#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_DIM 32

// CPU Matrix Multiplication (Sequential)
void cpuMatrixMul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0; // Use double for better precision during validation
            for (int k = 0; k < N; ++k) {
                sum += (double)A[i * N + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

// CPU Matrix Exponentiation (Binary Exponentiation)
void cpuMatrixExp(const float* A, float* C, int N, int power) {
    // Allocate temp matrices
    float* temp_base = (float*)malloc(N * N * sizeof(float));
    float* temp_acc = (float*)malloc(N * N * sizeof(float));
    float* temp_mul = (float*)malloc(N * N * sizeof(float));

    // Initialize temp_base = A
    memcpy(temp_base, A, N * N * sizeof(float));

    // Initialize temp_acc as Identity matrix
    memset(temp_acc, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        temp_acc[i * N + i] = 1.0f;
    }

    int p = power;
    while (p > 0) {
        if (p & 1) {
            cpuMatrixMul(temp_acc, temp_base, temp_mul, N);
            memcpy(temp_acc, temp_mul, N * N * sizeof(float));
        }
        if (p > 1) {
            cpuMatrixMul(temp_base, temp_base, temp_mul, N);
            memcpy(temp_base, temp_mul, N * N * sizeof(float));
        }
        p >>= 1;
    }

    memcpy(C, temp_acc, N * N * sizeof(float));

    free(temp_base);
    free(temp_acc);
    free(temp_mul);
}

// Naive GPU Matrix Multiplication Kernel
__global__ void gpuNaiveMatrixMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled GPU Matrix Multiplication Kernel using Shared Memory
__global__ void gpuTiledMatrixMul(const float* A, const float* B, float* C, int N) {
    __shared__ float sh_A[TILE_DIM][TILE_DIM];
    __shared__ float sh_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load tile from A into shared memory
        if (row < N && (m * TILE_DIM + tx) < N) {
            sh_A[ty][tx] = A[row * N + m * TILE_DIM + tx];
        } else {
            sh_A[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if (col < N && (m * TILE_DIM + ty) < N) {
            sh_B[ty][tx] = B[(m * TILE_DIM + ty) * N + col];
        } else {
            sh_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to launch matrix multiplication kernel
void launchMatrixMul(const float* d_A, const float* d_B, float* d_C, int N, bool use_tiled) {
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    if (use_tiled) {
        gpuTiledMatrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else {
        gpuNaiveMatrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
}

// GPU Binary Exponentiation
void gpuMatrixExp(const float* d_A, float* d_C, int N, int power, bool use_tiled) {
    float *d_temp_base, *d_temp_acc, *d_temp_mul;
    cudaMalloc(&d_temp_base, N * N * sizeof(float));
    cudaMalloc(&d_temp_acc, N * N * sizeof(float));
    cudaMalloc(&d_temp_mul, N * N * sizeof(float));

    // Initialize d_temp_base = d_A
    cudaMemcpy(d_temp_base, d_A, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Initialize d_temp_acc as Identity Matrix on Host, then copy to Device
    float* h_I = (float*)malloc(N * N * sizeof(float));
    memset(h_I, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_I[i * N + i] = 1.0f;
    }
    cudaMemcpy(d_temp_acc, h_I, N * N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_I);

    int p = power;
    while (p > 0) {
        if (p & 1) {
            launchMatrixMul(d_temp_acc, d_temp_base, d_temp_mul, N, use_tiled);
            // Copy result to d_temp_acc
            cudaMemcpy(d_temp_acc, d_temp_mul, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        if (p > 1) {
            launchMatrixMul(d_temp_base, d_temp_base, d_temp_mul, N, use_tiled);
            // Copy result to d_temp_base
            cudaMemcpy(d_temp_base, d_temp_mul, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        p >>= 1;
    }

    // Copy final result to output
    cudaMemcpy(d_C, d_temp_acc, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_temp_base);
    cudaFree(d_temp_acc);
    cudaFree(d_temp_mul);
}

// Validation function
bool validateResults(const float* cpu_res, const float* gpu_res, int N, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float diff = std::abs(cpu_res[i] - gpu_res[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "Max Absolute Difference: " << max_diff << " (Tolerance: " << tolerance << ")\n";
    return max_diff < tolerance;
}

int main(int argc, char* argv[]) {
    int N = 512; // Default size (small for safety/quick feedback, can scale up)
    int power = 100;

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        power = atoi(argv[2]);
    }

    std::cout << "========================================\n";
    std::cout << "Matrix Exponentiation: A^" << power << " for " << N << "x" << N << " Matrix\n";
    std::cout << "========================================\n";

    size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    float* h_A = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
    float* h_C_gpu_naive = (float*)malloc(bytes);
    float* h_C_gpu_tiled = (float*)malloc(bytes);

    // Initialize Matrix A with random values scaled by N to ensure spectral radius is around 1.0,
    // which prevents exponential explosion or decay to zero, making validation meaningful.
    srand(42);
    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = ((float)rand() / RAND_MAX);
            row_sum += h_A[i * N + j];
        }
        // Normalize rows to sum to 1.0
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] /= row_sum;
        }
    }

    // Device memory allocation
    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // 1. CPU Execution (only run for N <= 512 because N=1024 or higher is too slow for 100 power on single core CPU)
    if (N <= 512) {
        std::cout << "Running on CPU..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cpuMatrixExp(h_A, h_C_cpu, N, power);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "CPU Time: " << duration.count() << " ms\n\n";
    } else {
        std::cout << "Skipping CPU execution for large N (" << N << ") to avoid long wait times.\n\n";
    }

    // 2. GPU Naive Execution
    std::cout << "Running GPU Naive Exponentiation..." << std::endl;
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive);
    gpuMatrixExp(d_A, d_C, N, power, false);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);

    float milliseconds_naive = 0;
    cudaEventElapsedTime(&milliseconds_naive, start_naive, stop_naive);
    cudaMemcpy(h_C_gpu_naive, d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "GPU Naive Time: " << milliseconds_naive << " ms\n\n";

    // 3. GPU Tiled Execution
    std::cout << "Running GPU Tiled Exponentiation..." << std::endl;
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_tiled);
    gpuMatrixExp(d_A, d_C, N, power, true);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);

    float milliseconds_tiled = 0;
    cudaEventElapsedTime(&milliseconds_tiled, start_tiled, stop_tiled);
    cudaMemcpy(h_C_gpu_tiled, d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "GPU Tiled Time: " << milliseconds_tiled << " ms\n\n";

    // Validation
    if (N <= 512) {
        std::cout << "Validating GPU Naive vs CPU: ";
        validateResults(h_C_cpu, h_C_gpu_naive, N);
        std::cout << "Validating GPU Tiled vs CPU: ";
        validateResults(h_C_cpu, h_C_gpu_tiled, N);
    }
    std::cout << "Validating GPU Tiled vs GPU Naive: ";
    validateResults(h_C_gpu_naive, h_C_gpu_tiled, N);

    // Print speedup
    if (N <= 512) {
        // Calculate CPU time from std::chrono
        // (will print comparison)
    }
    std::cout << "GPU Tiled Speedup over GPU Naive: " << milliseconds_naive / milliseconds_tiled << "x\n";

    // Clean up
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu_naive);
    free(h_C_gpu_tiled);
    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);

    return 0;
}
