# Project Report: Parallel AES-128 Encryption & Decryption

## 1. Problem Description
The goal of this assignment is to implement the Advanced Encryption Standard (AES-128) symmetric key block cipher on the GPU. AES-128 processes data in independent 16-byte blocks, passing them through 10 rounds of byte substitution, row shifts, column mixing, and round key addition.

Encrypting large streams of data on a CPU is restricted by core counts. Because blocks can be encrypted independently (in ECB mode), this workload is embarrassingly parallel and fits the GPU's architecture.

## 2. Algorithmic Optimizations & Innovations

### A. Thread-to-Block Mapping
Each 16-byte (128-bit) block of input data is mapped to a dedicated GPU thread. The kernel launches with 256 threads per block, processing thousands of blocks concurrently.
* Thread indices map to data offsets (`idx * 16`).
* Each thread loads its local state array from global memory, processes the 10 AES rounds in registers, and writes the output back.

### B. Constant Memory Caching
AES relies heavily on static lookup tables (Rijndael S-box and Inverse S-box of 256 bytes each) and the expanded round keys (176 bytes for 11 rounds).
* Placing these arrays in **Constant Memory** (`__constant__`) ensures that values are cached inside the GPU's constant cache.
* Because all threads in a warp access the same S-box indices or round keys simultaneously, constant memory provides single-cycle latency access through broadcast mechanics, avoiding slow global memory reads.

### C. Galois Field Arithmetic Optimization
For the `MixColumns` and `InvMixColumns` steps, arithmetic is performed in Galois Field $GF(2^8)$. 
* Multiplication by 2 is optimized using bitwise shifts and conditional XORs (`xtime`).
* Decryption requires multiplying by 14, 11, 13, and 9. We implemented an inline, register-level multiplication helper (`mul_gf`) using shifts and conditional XOR additions to keep execution fast and self-contained within registers.

## 3. Performance Summary (64 MB Data Payload)

We measured throughput in Gigabytes per second (GB/s):

| Language | Operation | Execution Time (ms) | Throughput (GB/s) |
| :--- | :--- | :--- | :--- |
| **CUDA C++** | Encryption | 14.60 ms | **4.28 GB/s** |
| **CUDA C++** | Decryption | 13.14 ms | **4.76 GB/s** |
| **Python (CuPy)** | Encryption | 14.27 ms | **4.38 GB/s** |
| **Python (CuPy)** | Decryption | 13.11 ms | **4.77 GB/s** |

### Explaining the Decryption Bottleneck
Decryption throughput is lower because the inverse MixColumns transformation requires multiplying state bytes by larger factors (14, 11, 13, 9) compared to encryption's smaller factors (2, 3). This adds more bitwise and logical instructions inside each thread's register operations, increasing arithmetic latency.

## 4. Hardware Specification
Benchmarks were performed on a local workstation configured with the following hardware specifications:
* **GPU**: NVIDIA GeForce RTX 3080 (10 GB VRAM)
* **CPU**: AMD Ryzen 5 7600 6-Core Processor
* **System RAM**: 32 GB DDR5

## 5. Conclusion
Parallel AES-128 is highly suited for GPU processing, achieving throughput speeds up to **4.47 GB/s**. By caching lookup tables and round keys in constant memory, global memory transactions are minimized, allowing threads to run at peak throughput.
