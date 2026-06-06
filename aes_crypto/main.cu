#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// AES S-Box and Inverse S-Box
__constant__ uint8_t d_sbox[256];
__constant__ uint8_t d_inv_sbox[256];
__constant__ uint8_t d_RoundKeys[176]; // 11 round keys of 16 bytes each for AES-128

static const uint8_t h_sbox[256] = {
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
};

static const uint8_t h_inv_sbox[256] = {
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
};

// Key Expansion constants
static const uint8_t Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// Key Expansion on CPU
void keyExpansion(const uint8_t* key, uint8_t* roundKeys) {
    uint8_t temp[4];
    memcpy(roundKeys, key, 16);

    int bytesGenerated = 16;
    int rconIter = 1;

    while (bytesGenerated < 176) {
        // Read last 4 bytes generated
        for (int i = 0; i < 4; ++i) {
            temp[i] = roundKeys[bytesGenerated - 4 + i];
        }

        // Perform RotWord & SubWord if at start of key size boundary
        if (bytesGenerated % 16 == 0) {
            // RotWord
            uint8_t k = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = k;

            // SubWord
            temp[0] = h_sbox[temp[0]];
            temp[1] = h_sbox[temp[1]];
            temp[2] = h_sbox[temp[2]];
            temp[3] = h_sbox[temp[3]];

            // XOR with Rcon
            temp[0] ^= Rcon[rconIter++];
        }

        // XOR with the byte 16 positions back
        for (int i = 0; i < 4; ++i) {
            roundKeys[bytesGenerated] = roundKeys[bytesGenerated - 16] ^ temp[i];
            bytesGenerated++;
        }
    }
}

// Galois Field Multiplication in GF(2^8) helpers
__device__ inline uint8_t xtime(uint8_t x) {
    return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}

__device__ inline uint8_t mul_gf(uint8_t x, uint8_t y) {
    uint8_t res = 0;
    uint8_t temp = x;
    while (y > 0) {
        if (y & 1) res ^= temp;
        temp = xtime(temp);
        y >>= 1;
    }
    return res;
}

// Device functions for AES rounds
__device__ void addRoundKey(uint8_t* state, const uint8_t* roundKey) {
    for (int i = 0; i < 16; ++i) {
        state[i] ^= roundKey[i];
    }
}

__device__ void subBytes(uint8_t* state) {
    for (int i = 0; i < 16; ++i) {
        state[i] = d_sbox[state[i]];
    }
}

__device__ void invSubBytes(uint8_t* state) {
    for (int i = 0; i < 16; ++i) {
        state[i] = d_inv_sbox[state[i]];
    }
}

__device__ void shiftRows(uint8_t* state) {
    uint8_t temp;
    // Row 1: Left shift by 1
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;

    // Row 2: Left shift by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: Left shift by 3 (or Right shift by 1)
    temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;
}

__device__ void invShiftRows(uint8_t* state) {
    uint8_t temp;
    // Row 1: Right shift by 1
    temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;

    // Row 2: Right shift by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: Right shift by 3 (or Left shift by 1)
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;
}

__device__ void mixColumns(uint8_t* state) {
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

__device__ void invMixColumns(uint8_t* state) {
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

// Parallel AES Encryption Kernel
__global__ void aesEncryptKernel(const uint8_t* input, uint8_t* output, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        uint8_t state[16];
        // Load block
        for (int i = 0; i < 16; ++i) {
            state[i] = input[idx * 16 + i];
        }

        // Initial Round
        addRoundKey(state, &d_RoundKeys[0]);

        // 9 Main Rounds
        for (int round = 1; round < 10; ++round) {
            subBytes(state);
            shiftRows(state);
            mixColumns(state);
            addRoundKey(state, &d_RoundKeys[round * 16]);
        }

        // Final Round
        subBytes(state);
        shiftRows(state);
        addRoundKey(state, &d_RoundKeys[10 * 16]);

        // Write block
        for (int i = 0; i < 16; ++i) {
            output[idx * 16 + i] = state[i];
        }
    }
}

// Parallel AES Decryption Kernel
__global__ void aesDecryptKernel(const uint8_t* input, uint8_t* output, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        uint8_t state[16];
        // Load block
        for (int i = 0; i < 16; ++i) {
            state[i] = input[idx * 16 + i];
        }

        // Initial Round
        addRoundKey(state, &d_RoundKeys[10 * 16]);
        invShiftRows(state);
        invSubBytes(state);

        // 9 Main Rounds (in reverse order)
        for (int round = 9; round > 0; --round) {
            addRoundKey(state, &d_RoundKeys[round * 16]);
            invMixColumns(state);
            invShiftRows(state);
            invSubBytes(state);
        }

        // Final Round
        addRoundKey(state, &d_RoundKeys[0]);

        // Write block
        for (int i = 0; i < 16; ++i) {
            output[idx * 16 + i] = state[i];
        }
    }
}

int main(int argc, char* argv[]) {
    int dataSize = 64 * 1024 * 1024; // 64 MB default
    if (argc > 1) {
        dataSize = atoi(argv[1]) * 1024 * 1024;
    }

    int numBlocks = dataSize / 16;

    std::cout << "========================================\n";
    std::cout << "Parallel AES-128 (ECB Mode) Encryption/Decryption\n";
    std::cout << "Data Size: " << dataSize / (1024 * 1024) << " MB (" << numBlocks << " blocks)\n";
    std::cout << "========================================\n";

    // Setup AES key and expand it
    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    uint8_t roundKeys[176];
    keyExpansion(key, roundKeys);

    // Initialize Host buffers
    uint8_t* h_plaintext = (uint8_t*)malloc(dataSize);
    uint8_t* h_ciphertext = (uint8_t*)malloc(dataSize);
    uint8_t* h_decrypted = (uint8_t*)malloc(dataSize);

    // Populate plaintext with random bytes
    srand(42);
    for (int i = 0; i < dataSize; ++i) {
        h_plaintext[i] = rand() % 256;
    }

    // Copy lookup tables and round keys to Constant Memory
    cudaMemcpyToSymbol(d_sbox, h_sbox, 256);
    cudaMemcpyToSymbol(d_inv_sbox, h_inv_sbox, 256);
    cudaMemcpyToSymbol(d_RoundKeys, roundKeys, 176);

    // Allocate GPU buffers
    uint8_t *d_plaintext, *d_ciphertext, *d_decrypted;
    cudaMalloc(&d_plaintext, dataSize);
    cudaMalloc(&d_ciphertext, dataSize);
    cudaMalloc(&d_decrypted, dataSize);

    // Copy plain text to device
    cudaMemcpy(d_plaintext, h_plaintext, dataSize, cudaMemcpyHostToDevice);

    // Run Encryption
    int threadsPerBlock = 256;
    int numThreadBlocks = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_enc, stop_enc;
    cudaEventCreate(&start_enc);
    cudaEventCreate(&stop_enc);

    std::cout << "Running GPU Encryption..." << std::endl;
    cudaEventRecord(start_enc);
    aesEncryptKernel<<<numThreadBlocks, threadsPerBlock>>>(d_plaintext, d_ciphertext, numBlocks);
    cudaEventRecord(stop_enc);
    cudaEventSynchronize(stop_enc);

    float ms_enc = 0;
    cudaEventElapsedTime(&ms_enc, start_enc, stop_enc);
    std::cout << "Encryption Time: " << ms_enc << " ms\n";
    std::cout << "Encryption Throughput: " << (double)dataSize / (ms_enc / 1000.0) / (1024.0 * 1024.0 * 1024.0) << " GB/s\n\n";

    // Run Decryption
    cudaEvent_t start_dec, stop_dec;
    cudaEventCreate(&start_dec);
    cudaEventCreate(&stop_dec);

    std::cout << "Running GPU Decryption..." << std::endl;
    cudaEventRecord(start_dec);
    aesDecryptKernel<<<numThreadBlocks, threadsPerBlock>>>(d_ciphertext, d_decrypted, numBlocks);
    cudaEventRecord(stop_dec);
    cudaEventSynchronize(stop_dec);

    float ms_dec = 0;
    cudaEventElapsedTime(&ms_dec, start_dec, stop_dec);
    std::cout << "Decryption Time: " << ms_dec << " ms\n";
    std::cout << "Decryption Throughput: " << (double)dataSize / (ms_dec / 1000.0) / (1024.0 * 1024.0 * 1024.0) << " GB/s\n\n";

    // Copy back and Validate
    cudaMemcpy(h_decrypted, d_decrypted, dataSize, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < dataSize; ++i) {
        if (h_plaintext[i] != h_decrypted[i]) {
            std::cout << "Mismatch at byte " << i << "! Expected: " << (int)h_plaintext[i] << ", Got: " << (int)h_decrypted[i] << "\n";
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "SUCCESS: Decrypted plaintext matches original plaintext perfectly!\n";
    }

    // Clean up
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_decrypted);
    free(h_plaintext);
    free(h_ciphertext);
    free(h_decrypted);
    cudaEventDestroy(start_enc);
    cudaEventDestroy(stop_enc);
    cudaEventDestroy(start_dec);
    cudaEventDestroy(stop_dec);

    return 0;
}
