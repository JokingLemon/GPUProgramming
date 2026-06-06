#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BLOCK_DIM 16

// CPU 2D Heat Diffusion Stencil Solver (5-point stencil)
void cpuHeatSolver(const float* current, float* next, int W, int H, float c) {
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            next[idx] = current[idx] + c * (
                current[idx + 1] +       // East
                current[idx - 1] +       // West
                current[idx + W] +       // South
                current[idx - W] -       // North
                4.0f * current[idx]
            );
        }
    }
}

// Naive GPU Stencil Kernel
__global__ void gpuNaiveStencil(const float* current, float* next, int W, int H, float c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < W - 1 && y > 0 && y < H - 1) {
        int idx = y * W + x;
        next[idx] = current[idx] + c * (
            current[idx + 1] +
            current[idx - 1] +
            current[idx + W] +
            current[idx - W] -
            4.0f * current[idx]
        );
    }
}

// Shared Memory GPU Stencil Kernel (handling 1-pixel wide halos)
__global__ void gpuSharedStencil(const float* current, float* next, int W, int H, float c) {
    // Shared memory size is (BLOCK_DIM + 2) x (BLOCK_DIM + 2)
    __shared__ float sh_tile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // Shift shared indices by 1 to accommodate top/left halos
    int sh_x = tx + 1;
    int sh_y = ty + 1;

    // Clamp coordinates to grid boundaries
    int x_clamped = min(max(gx, 0), W - 1);
    int y_clamped = min(max(gy, 0), H - 1);

    // 1. Load center element
    sh_tile[sh_y][sh_x] = current[y_clamped * W + x_clamped];

    // 2. Load halos
    // Left halo
    if (tx == 0) {
        sh_tile[sh_y][0] = current[y_clamped * W + max(gx - 1, 0)];
    }
    // Right halo
    if (tx == BLOCK_DIM - 1 || gx == W - 1) {
        sh_tile[sh_y][BLOCK_DIM + 1] = current[y_clamped * W + min(gx + 1, W - 1)];
    }
    // Top halo
    if (ty == 0) {
        sh_tile[0][sh_x] = current[max(gy - 1, 0) * W + x_clamped];
    }
    // Bottom halo
    if (ty == BLOCK_DIM - 1 || gy == H - 1) {
        sh_tile[BLOCK_DIM + 1][sh_x] = current[min(gy + 1, H - 1) * W + x_clamped];
    }

    __syncthreads();

    // 3. Compute stencil for interior cells only
    if (gx > 0 && gx < W - 1 && gy > 0 && gy < H - 1) {
        next[gy * W + gx] = sh_tile[sh_y][sh_x] + c * (
            sh_tile[sh_y][sh_x + 1] + // East
            sh_tile[sh_y][sh_x - 1] + // West
            sh_tile[sh_y + 1][sh_x] + // South
            sh_tile[sh_y - 1][sh_x] - // North
            4.0f * sh_tile[sh_y][sh_x]
        );
    }
}

// Validation helper
bool validateGrid(const float* cpu_grid, const float* gpu_grid, int W, int H, float tolerance = 1e-4f) {
    float max_diff = 0.0f;
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            float diff = std::abs(cpu_grid[idx] - gpu_grid[idx]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    std::cout << "Max Absolute Difference: " << max_diff << " (Tolerance: " << tolerance << ")\n";
    return max_diff < tolerance;
}

int main(int argc, char* argv[]) {
    int W = 1024;
    int H = 1024;
    int steps = 1000;
    float c = 0.1f; // Diffusion coefficient

    if (argc > 1) W = atoi(argv[1]);
    if (argc > 2) H = atoi(argv[2]);
    if (argc > 3) steps = atoi(argv[3]);

    std::cout << "========================================\n";
    std::cout << "2D PDE Heat Solver: Grid " << W << "x" << H << ", Steps: " << steps << "\n";
    std::cout << "========================================\n";

    size_t bytes = W * H * sizeof(float);

    // Host memory
    float* h_grid = (float*)malloc(bytes);
    float* h_grid_cpu = (float*)malloc(bytes);
    float* h_grid_gpu = (float*)malloc(bytes);

    // Initialize boundary conditions: Heat sources at boundaries
    memset(h_grid, 0, bytes);
    for (int x = 0; x < W; ++x) {
        h_grid[x] = 100.0f;             // Top border hot (100C)
        h_grid[(H - 1) * W + x] = 50.0f; // Bottom border warm (50C)
    }
    for (int y = 0; y < H; ++y) {
        h_grid[y * W] = 75.0f;          // Left border warm (75C)
        h_grid[y * W + (W - 1)] = 0.0f; // Right border cold (0C)
    }

    // Copy to CPU grid buffers
    memcpy(h_grid_cpu, h_grid, bytes);
    float* h_grid_cpu_next = (float*)malloc(bytes);
    memcpy(h_grid_cpu_next, h_grid, bytes);

    // Device grids (Double Buffering)
    float *d_gridA, *d_gridB;
    cudaMalloc(&d_gridA, bytes);
    cudaMalloc(&d_gridB, bytes);

    // 1. CPU Solve
    std::cout << "Running CPU Solver..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        cpuHeatSolver(h_grid_cpu, h_grid_cpu_next, W, H, c);
        std::swap(h_grid_cpu, h_grid_cpu_next);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end - start;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms\n\n";

    // 2. GPU Naive Solve
    std::cout << "Running GPU Naive Stencil..." << std::endl;
    cudaMemcpy(d_gridA, h_grid, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridB, h_grid, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((W + BLOCK_DIM - 1) / BLOCK_DIM, (H + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive);
    float* curr = d_gridA;
    float* next = d_gridB;
    for (int s = 0; s < steps; ++s) {
        gpuNaiveStencil<<<numBlocks, threadsPerBlock>>>(curr, next, W, H, c);
        std::swap(curr, next);
    }
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start_naive, stop_naive);
    cudaMemcpy(h_grid_gpu, curr, bytes, cudaMemcpyDeviceToHost);
    std::cout << "GPU Naive Time: " << ms_naive << " ms\n";
    std::cout << "Validation (Naive vs CPU): ";
    validateGrid(h_grid_cpu, h_grid_gpu, W, H);
    std::cout << std::endl;

    // 3. GPU Shared Solve
    std::cout << "Running GPU Shared Memory Stencil..." << std::endl;
    cudaMemcpy(d_gridA, h_grid, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridB, h_grid, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);

    cudaEventRecord(start_shared);
    curr = d_gridA;
    next = d_gridB;
    for (int s = 0; s < steps; ++s) {
        gpuSharedStencil<<<numBlocks, threadsPerBlock>>>(curr, next, W, H, c);
        std::swap(curr, next);
    }
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);

    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start_shared, stop_shared);
    cudaMemcpy(h_grid_gpu, curr, bytes, cudaMemcpyDeviceToHost);
    std::cout << "GPU Shared Time: " << ms_shared << " ms\n";
    std::cout << "Validation (Shared vs CPU): ";
    validateGrid(h_grid_cpu, h_grid_gpu, W, H);
    std::cout << std::endl;

    std::cout << "GPU Shared Speedup over GPU Naive: " << ms_naive / ms_shared << "x\n";

    // Cleanup
    cudaFree(d_gridA);
    cudaFree(d_gridB);
    free(h_grid);
    free(h_grid_cpu);
    free(h_grid_cpu_next);
    free(h_grid_gpu);
    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);

    return 0;
}
