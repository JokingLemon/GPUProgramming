import time
import numpy as np
import cupy as cp

# S-Box, Inv S-Box, and Key constants for raw CUDA kernel compilation
AES_CUDA_CODE = r'''
typedef unsigned char uint8_t;

__constant__ uint8_t d_sbox[256];
__constant__ uint8_t d_inv_sbox[256];
__constant__ uint8_t d_RoundKeys[176];

extern "C" __device__ inline uint8_t xtime(uint8_t x) {
    return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}

extern "C" __device__ inline uint8_t mul_gf(uint8_t x, uint8_t y) {
    uint8_t res = 0;
    uint8_t temp = x;
    while (y > 0) {
        if (y & 1) res ^= temp;
        temp = xtime(temp);
        y >>= 1;
    }
    return res;
}

extern "C" __device__ void addRoundKey(uint8_t* state, const uint8_t* roundKey) {
    for (int i = 0; i < 16; ++i) {
        state[i] ^= roundKey[i];
    }
}

extern "C" __device__ void subBytes(uint8_t* state) {
    for (int i = 0; i < 16; ++i) {
        state[i] = d_sbox[state[i]];
    }
}

extern "C" __device__ void invSubBytes(uint8_t* state) {
    for (int i = 0; i < 16; ++i) {
        state[i] = d_inv_sbox[state[i]];
    }
}

extern "C" __device__ void shiftRows(uint8_t* state) {
    uint8_t temp;
    temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    temp = state[13]; state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = temp;
}

extern "C" __device__ void invShiftRows(uint8_t* state) {
    uint8_t temp;
    temp = state[13]; state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = temp;
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
}

extern "C" __device__ void mixColumns(uint8_t* state) {
    for (int i = 0; i < 4; ++i) {
        int col = i * 4;
        uint8_t s0 = state[col];
        uint8_t s1 = state[col + 1];
        uint8_t s2 = state[col + 2];
        uint8_t s3 = state[col + 3];
        state[col]     = xtime(s0) ^ (xtime(s1) ^ s1) ^ s2 ^ s3;
        state[col + 1] = s0 ^ xtime(s1) ^ (xtime(s2) ^ s2) ^ s3;
        state[col + 2] = s0 ^ s1 ^ xtime(s2) ^ (xtime(s3) ^ s3);
        state[col + 3] = (xtime(s0) ^ s0) ^ s1 ^ s2 ^ xtime(s3);
    }
}

extern "C" __device__ void invMixColumns(uint8_t* state) {
    for (int i = 0; i < 4; ++i) {
        int col = i * 4;
        uint8_t s0 = state[col];
        uint8_t s1 = state[col + 1];
        uint8_t s2 = state[col + 2];
        uint8_t s3 = state[col + 3];
        state[col]     = mul_gf(s0, 0x0e) ^ mul_gf(s1, 0x0b) ^ mul_gf(s2, 0x0d) ^ mul_gf(s3, 0x09);
        state[col + 1] = mul_gf(s0, 0x09) ^ mul_gf(s1, 0x0e) ^ mul_gf(s2, 0x0b) ^ mul_gf(s3, 0x0d);
        state[col + 2] = mul_gf(s0, 0x0d) ^ mul_gf(s1, 0x09) ^ mul_gf(s2, 0x0e) ^ mul_gf(s3, 0x0b);
        state[col + 3] = mul_gf(s0, 0x0b) ^ mul_gf(s1, 0x0d) ^ mul_gf(s2, 0x09) ^ mul_gf(s3, 0x0e);
    }
}

extern "C" __global__ void aesEncryptKernel(const uint8_t* input, uint8_t* output, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        uint8_t state[16];
        for (int i = 0; i < 16; ++i) state[i] = input[idx * 16 + i];
        addRoundKey(state, &d_RoundKeys[0]);
        for (int round = 1; round < 10; ++round) {
            subBytes(state);
            shiftRows(state);
            mixColumns(state);
            addRoundKey(state, &d_RoundKeys[round * 16]);
        }
        subBytes(state);
        shiftRows(state);
        addRoundKey(state, &d_RoundKeys[10 * 16]);
        for (int i = 0; i < 16; ++i) output[idx * 16 + i] = state[i];
    }
}

extern "C" __global__ void aesDecryptKernel(const uint8_t* input, uint8_t* output, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        uint8_t state[16];
        for (int i = 0; i < 16; ++i) state[i] = input[idx * 16 + i];
        addRoundKey(state, &d_RoundKeys[10 * 16]);
        invShiftRows(state);
        invSubBytes(state);
        for (int round = 9; round > 0; --round) {
            addRoundKey(state, &d_RoundKeys[round * 16]);
            invMixColumns(state);
            invShiftRows(state);
            invSubBytes(state);
        }
        addRoundKey(state, &d_RoundKeys[0]);
        for (int i = 0; i < 16; ++i) output[idx * 16 + i] = state[i];
    }
}
'''

h_sbox = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

h_inv_sbox = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
], dtype=np.uint8)

Rcon = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

def key_expansion(key):
    round_keys = np.zeros(176, dtype=np.uint8)
    round_keys[:16] = key
    
    bytes_generated = 16
    rcon_iter = 1
    
    while bytes_generated < 176:
        temp = round_keys[bytes_generated - 4:bytes_generated].copy()
        if bytes_generated % 16 == 0:
            # RotWord
            temp = np.roll(temp, -1)
            # SubWord
            temp = h_sbox[temp]
            # XOR with Rcon
            temp[0] ^= Rcon[rcon_iter]
            rcon_iter += 1
            
        round_keys[bytes_generated:bytes_generated+4] = round_keys[bytes_generated-16:bytes_generated-12] ^ temp
        bytes_generated += 4
        
    return round_keys

def main():
    data_size = 64 * 1024 * 1024 # 64 MB
    num_blocks = data_size // 16

    print("========================================")
    print(f"Python Parallel AES-128 (ECB Mode)")
    print(f"Data Size: {data_size // (1024 * 1024)} MB ({num_blocks} blocks)")
    print("========================================")

    # Key Expansion
    key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)
    round_keys = key_expansion(key)

    # Compile module in CuPy
    module = cp.RawModule(code=AES_CUDA_CODE)
    
    # Get pointer to constant memories
    d_sbox_ptr = module.get_global('d_sbox')
    d_inv_sbox_ptr = module.get_global('d_inv_sbox')
    d_RoundKeys_ptr = module.get_global('d_RoundKeys')

    # Copy lookup tables to constant memory on GPU
    d_sbox_mem = cp.ndarray(256, dtype=cp.uint8, memptr=d_sbox_ptr)
    d_sbox_mem.set(h_sbox)
    
    d_inv_sbox_mem = cp.ndarray(256, dtype=cp.uint8, memptr=d_inv_sbox_ptr)
    d_inv_sbox_mem.set(h_inv_sbox)
    
    d_roundkeys_mem = cp.ndarray(176, dtype=cp.uint8, memptr=d_RoundKeys_ptr)
    d_roundkeys_mem.set(round_keys)

    # Retrieve kernels
    encrypt_kernel = module.get_function('aesEncryptKernel')
    decrypt_kernel = module.get_function('aesDecryptKernel')

    # Generate random plaintext
    np.random.seed(42)
    h_plaintext = np.random.randint(0, 256, size=data_size, dtype=np.uint8)
    
    # Allocate device arrays
    d_plaintext = cp.asarray(h_plaintext)
    d_ciphertext = cp.empty_like(d_plaintext)
    d_decrypted = cp.empty_like(d_plaintext)

    threads_per_block = 256
    blocks_per_grid = (num_blocks + threads_per_block - 1) // threads_per_block

    # Warmup
    encrypt_kernel((1,), (threads_per_block,), (d_plaintext, d_ciphertext, np.int32(16)))
    decrypt_kernel((1,), (threads_per_block,), (d_ciphertext, d_decrypted, np.int32(16)))
    cp.cuda.Stream.null.synchronize()

    # 1. Measure GPU Encryption
    start = time.perf_counter()
    encrypt_kernel((blocks_per_grid,), (threads_per_block,), (d_plaintext, d_ciphertext, np.int32(num_blocks)))
    cp.cuda.Stream.null.synchronize()
    enc_time = (time.perf_counter() - start) * 1000.0
    print(f"CuPy Encryption Time: {enc_time:.2f} ms")
    print(f"CuPy Encryption Throughput: {data_size / (enc_time / 1000.0) / (1024**3):.2f} GB/s")

    # 2. Measure GPU Decryption
    start = time.perf_counter()
    decrypt_kernel((blocks_per_grid,), (threads_per_block,), (d_ciphertext, d_decrypted, np.int32(num_blocks)))
    cp.cuda.Stream.null.synchronize()
    dec_time = (time.perf_counter() - start) * 1000.0
    print(f"\nCuPy Decryption Time: {dec_time:.2f} ms")
    print(f"CuPy Decryption Throughput: {data_size / (dec_time / 1000.0) / (1024**3):.2f} GB/s")

    # Validate
    h_decrypted = cp.asnumpy(d_decrypted)
    if np.array_equal(h_plaintext, h_decrypted):
        print("\nSUCCESS: Decrypted plaintext matches original plaintext perfectly in Python!")
    else:
        print("\nFAIL: Mismatch found between original and decrypted plaintext!")

if __name__ == '__main__':
    main()
