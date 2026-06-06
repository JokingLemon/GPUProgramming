import time
import numpy as np
import cupy as cp

TILE_DIM = 32

# Naive CUDA kernel compiled via CuPy RawKernel
naive_kernel = cp.RawKernel(r'''
extern "C" __global__
void gpuNaiveMatrixMul(const float* A, const float* B, float* C, int N) {
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
''', 'gpuNaiveMatrixMul')

# Tiled CUDA kernel compiled via CuPy RawKernel
tiled_kernel = cp.RawKernel(r'''
extern "C" __global__
void gpuTiledMatrixMul(const float* A, const float* B, float* C, int N) {
    __shared__ float sh_A[32][32];
    __shared__ float sh_B[32][32];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    float sum = 0.0f;

    for (int m = 0; m < (N + 32 - 1) / 32; ++m) {
        if (row < N && (m * 32 + tx) < N) {
            sh_A[ty][tx] = A[row * N + m * 32 + tx];
        } else {
            sh_A[ty][tx] = 0.0f;
        }

        if (col < N && (m * 32 + ty) < N) {
            sh_B[ty][tx] = B[(m * 32 + ty) * N + col];
        } else {
            sh_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
''', 'gpuTiledMatrixMul')

def launch_cupy_kernel(kernel_func, d_A, d_B, d_C, N):
    grid_size = ((N + TILE_DIM - 1) // TILE_DIM, (N + TILE_DIM - 1) // TILE_DIM)
    block_size = (TILE_DIM, TILE_DIM)
    kernel_func(grid_size, block_size, (d_A, d_B, d_C, np.int32(N)))

# GPU Binary Exponentiation using CuPy RawKernels
def gpu_matrix_exp_cupy(d_A, N, power, kernel_func):
    d_temp_base = d_A.copy()
    d_temp_acc = cp.eye(N, dtype=cp.float32)
    d_temp_mul = cp.empty((N, N), dtype=cp.float32)

    p = power
    while p > 0:
        if p & 1:
            launch_cupy_kernel(kernel_func, d_temp_acc, d_temp_base, d_temp_mul, N)
            d_temp_acc = d_temp_mul.copy()
        if p > 1:
            launch_cupy_kernel(kernel_func, d_temp_base, d_temp_base, d_temp_mul, N)
            d_temp_base = d_temp_mul.copy()
        p >>= 1
    
    return d_temp_acc

def main():
    N = 512
    power = 100
    print("========================================")
    print(f"Python Matrix Exponentiation: A^{power} for {N}x{N} Matrix")
    print("========================================")

    # Initialize matrix A and normalize rows to keep values bounded
    np.random.seed(42)
    h_A = np.random.rand(N, N).astype(np.float32)
    for i in range(N):
        h_A[i] /= np.sum(h_A[i])

    # 1. CPU Benchmark (NumPy)
    start = time.perf_counter()
    cpu_res = np.linalg.matrix_power(h_A, power)
    cpu_time = (time.perf_counter() - start) * 1000.0
    print(f"NumPy CPU Time: {cpu_time:.2f} ms")

    # Move data to GPU using CuPy
    d_A = cp.asarray(h_A)

    # Warmup kernels to compile and cache them
    d_warm_out = cp.empty((N, N), dtype=cp.float32)
    launch_cupy_kernel(naive_kernel, d_A, d_A, d_warm_out, N)
    launch_cupy_kernel(tiled_kernel, d_A, d_A, d_warm_out, N)
    cp.cuda.Stream.null.synchronize()

    # 2. Custom CuPy Naive GPU Exponentiation
    start = time.perf_counter()
    gpu_naive_res = gpu_matrix_exp_cupy(d_A, N, power, naive_kernel)
    cp.cuda.Stream.null.synchronize()
    gpu_naive_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Raw Naive Time: {gpu_naive_time:.2f} ms")

    # 3. Custom CuPy Tiled GPU Exponentiation
    start = time.perf_counter()
    gpu_tiled_res = gpu_matrix_exp_cupy(d_A, N, power, tiled_kernel)
    cp.cuda.Stream.null.synchronize()
    gpu_tiled_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Raw Tiled Time: {gpu_tiled_time:.2f} ms")

    # 4. CuPy Built-in Matrix Power (compiled library baseline)
    start = time.perf_counter()
    cupy_builtin_res = cp.linalg.matrix_power(d_A, power)
    cp.cuda.Stream.null.synchronize()
    cupy_builtin_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Built-in Time: {cupy_builtin_time:.2f} ms")

    # Verification
    cpu_res_gpu = cp.asarray(cpu_res)
    max_diff_naive = float(cp.max(cp.abs(cpu_res_gpu - gpu_naive_res)))
    max_diff_tiled = float(cp.max(cp.abs(cpu_res_gpu - gpu_tiled_res)))
    max_diff_builtin = float(cp.max(cp.abs(cpu_res_gpu - cupy_builtin_res)))

    print(f"\nVerification:")
    print(f"Max Diff (CuPy Raw Naive vs CPU): {max_diff_naive:.2e}")
    print(f"Max Diff (CuPy Raw Tiled vs CPU): {max_diff_tiled:.2e}")
    print(f"Max Diff (CuPy Built-in vs CPU): {max_diff_builtin:.2e}")
    
    print(f"\nSpeedups:")
    print(f"CuPy Raw Tiled Speedup over CuPy Raw Naive: {gpu_naive_time / gpu_tiled_time:.2f}x")
    print(f"CuPy Built-in Speedup over CuPy Raw Tiled: {gpu_tiled_time / cupy_builtin_time:.2f}x")

if __name__ == '__main__':
    main()
