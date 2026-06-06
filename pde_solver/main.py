import time
import numpy as np
import cupy as cp

BLOCK_DIM = 16

# Naive CUDA kernel compiled via CuPy RawKernel
naive_kernel = cp.RawKernel(r'''
extern "C" __global__
void gpuNaiveStencil(const float* current, float* next, int W, int H, float c) {
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
''', 'gpuNaiveStencil')

# Shared Memory CUDA kernel compiled via CuPy RawKernel
shared_kernel = cp.RawKernel(r'''
extern "C" __global__
void gpuSharedStencil(const float* current, float* next, int W, int H, float c) {
    __shared__ float sh_tile[16 + 2][16 + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int sh_x = tx + 1;
    int sh_y = ty + 1;

    int x_clamped = min(max(gx, 0), W - 1);
    int y_clamped = min(max(gy, 0), H - 1);

    sh_tile[sh_y][sh_x] = current[y_clamped * W + x_clamped];

    if (tx == 0) {
        sh_tile[sh_y][0] = current[y_clamped * W + max(gx - 1, 0)];
    }
    if (tx == 16 - 1 || gx == W - 1) {
        sh_tile[sh_y][16 + 1] = current[y_clamped * W + min(gx + 1, W - 1)];
    }
    if (ty == 0) {
        sh_tile[0][sh_x] = current[max(gy - 1, 0) * W + x_clamped];
    }
    if (ty == 16 - 1 || gy == H - 1) {
        sh_tile[16 + 1][sh_x] = current[min(gy + 1, H - 1) * W + x_clamped];
    }

    __syncthreads();

    if (gx > 0 && gx < W - 1 && gy > 0 && gy < H - 1) {
        next[gy * W + gx] = sh_tile[sh_y][sh_x] + c * (
            sh_tile[sh_y][sh_x + 1] +
            sh_tile[sh_y][sh_x - 1] +
            sh_tile[sh_y + 1][sh_x] +
            sh_tile[sh_y - 1][sh_x] -
            4.0f * sh_tile[sh_y][sh_x]
        );
    }
}
''', 'gpuSharedStencil')

# NumPy CPU solver (using slicing for high performance in Python)
def cpu_numpy_solve(grid, steps, c):
    curr = grid.copy()
    for _ in range(steps):
        # Apply 5-point stencil vectorized
        next_grid = curr.copy()
        next_grid[1:-1, 1:-1] = curr[1:-1, 1:-1] + c * (
            curr[1:-1, 2:] +      # East
            curr[1:-1, :-2] +     # West
            curr[2:, 1:-1] +      # South
            curr[:-2, 1:-1] -      # North
            4.0 * curr[1:-1, 1:-1]
        )
        curr = next_grid
    return curr

def run_gpu_solver(kernel, d_grid, W, H, steps, c):
    d_curr = d_grid.copy()
    d_next = d_grid.copy()
    
    grid_size = ((W + BLOCK_DIM - 1) // BLOCK_DIM, (H + BLOCK_DIM - 1) // BLOCK_DIM)
    block_size = (BLOCK_DIM, BLOCK_DIM)
    
    for _ in range(steps):
        kernel(grid_size, block_size, (d_curr, d_next, np.int32(W), np.int32(H), np.float32(c)))
        d_curr, d_next = d_next, d_curr
        
    return d_curr

def main():
    W = 1024
    H = 1024
    steps = 1000
    c = 0.1

    print("========================================")
    print(f"Python 2D PDE Heat Solver: Grid {W}x{H}, Steps: {steps}")
    print("========================================")

    # Initialize Grid on CPU
    h_grid = np.zeros((H, W), dtype=np.float32)
    # Boundary conditions
    h_grid[0, :] = 100.0   # Top hot
    h_grid[-1, :] = 50.0   # Bottom warm
    h_grid[:, 0] = 75.0    # Left warm
    h_grid[:, -1] = 0.0    # Right cold

    # 1. NumPy CPU Solve
    start = time.perf_counter()
    cpu_res = cpu_numpy_solve(h_grid, steps, c)
    cpu_time = (time.perf_counter() - start) * 1000.0
    print(f"NumPy CPU (Vectorized) Time: {cpu_time:.2f} ms")

    # Transfer grid to GPU
    d_grid = cp.asarray(h_grid)

    # Warmup kernels
    d_warm_out = cp.empty((H, W), dtype=cp.float32)
    naive_kernel((1, 1), (BLOCK_DIM, BLOCK_DIM), (d_grid, d_warm_out, np.int32(W), np.int32(H), np.float32(c)))
    shared_kernel((1, 1), (BLOCK_DIM, BLOCK_DIM), (d_grid, d_warm_out, np.int32(W), np.int32(H), np.float32(c)))
    cp.cuda.Stream.null.synchronize()

    # 2. GPU Naive Solver
    start = time.perf_counter()
    gpu_naive_res = run_gpu_solver(naive_kernel, d_grid, W, H, steps, c)
    cp.cuda.Stream.null.synchronize()
    gpu_naive_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Raw Naive Time: {gpu_naive_time:.2f} ms")

    # 3. GPU Shared Solver
    start = time.perf_counter()
    gpu_shared_res = run_gpu_solver(shared_kernel, d_grid, W, H, steps, c)
    cp.cuda.Stream.null.synchronize()
    gpu_shared_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Raw Shared Time: {gpu_shared_time:.2f} ms")

    # Validation
    cpu_res_gpu = cp.asarray(cpu_res)
    max_diff_naive = float(cp.max(cp.abs(cpu_res_gpu[1:-1, 1:-1] - gpu_naive_res[1:-1, 1:-1])))
    max_diff_shared = float(cp.max(cp.abs(cpu_res_gpu[1:-1, 1:-1] - gpu_shared_res[1:-1, 1:-1])))

    print(f"\nVerification:")
    print(f"Max Diff (CuPy Raw Naive vs CPU): {max_diff_naive:.2e}")
    print(f"Max Diff (CuPy Raw Shared vs CPU): {max_diff_shared:.2e}")
    
    print(f"\nSpeedups:")
    print(f"CuPy Raw Shared Speedup over CuPy Raw Naive: {gpu_naive_time / gpu_shared_time:.2f}x")
    print(f"CuPy Raw Shared Speedup over NumPy CPU (Vectorized): {cpu_time / gpu_shared_time:.2f}x")

if __name__ == '__main__':
    main()
