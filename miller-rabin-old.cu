#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <limits>
#include <cuda.h>
typedef std::chrono::high_resolution_clock Clock;

#define NUM_TEST 10000000
#define NUM_BLOCKS 1
#define K 100

using namespace std;
  

// Helper function for modular exponentiation.
// Returns a^e (mode n)
__device__ unsigned long long modexp(unsigned long long a, unsigned long long e, unsigned long long n) {
    unsigned long long res = 1;
 
    a = a % n;  // Compute a mod n first (if a > n)
 
    while (e > 0) 
    {
        // exponent is odd
        if (e & 1) 
            res = (res * a) % n;
 
        // exponent is even
        e = e >> 1; // Shift right one (divide by 2)
        a = (a * a) % n;  // Set a = a^2 mod n
    }
    return res;
}
  
// Called each iteration of witness loop.
// Returns false if composite or true if probably prime
__global__ void witnessTest(float *d_random_nums, volatile bool *shared_result, unsigned long long d, unsigned long long n) {
    if (!(*shared_result)) return;
    // Pick a random number in [2..n-2]
    unsigned long long a = 2 + d_random_nums[threadIdx.x] * (n-4);
    unsigned long long x = modexp(a, d, n); 

    if (x == 1  || x == n-1) {
        return;
    }
  
    // Iterate r times (2^r * d = n - 1)
    while (d != n-1) { 
        x = (x * x) % n; 
        d *= 2; 
  
        if (x == 1) {
            *shared_result = false;
            return;
        } 
        if (x == n-1) {
            return;
        } 
    } 
  
    // Return composite 
    *shared_result = false;
}
  
// See: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
// Returns true if k-probably prime (k is a parameter that determines accuracy)
// Returns false if composite
bool millerRabinPrimalityTest(unsigned long long n, unsigned long long k, float *random_nums) {
    if (n == 4) return false; 
    if (n <= 3) return true; 
  
    // Find r such that n = 2^d * r + 1 for some r >= 1 
    unsigned long long d = n - 1; 
    while (d % 2 == 0) {
        d /= 2; 
    }

    volatile bool *d_result;
    float *d_random_nums;
    bool result = true;

    cudaMalloc((void **) &d_result, sizeof(bool));
    cudaMalloc((void **) &d_random_nums, K * sizeof(float));

    cudaMemcpy((void *) d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_random_nums, random_nums, K * sizeof(float), cudaMemcpyHostToDevice);

    // Witness loop to repeat k times
    // As long as K <= 256, run on 1 block
    witnessTest<<<NUM_BLOCKS, K>>>(d_random_nums, d_result, d, n);

    cudaMemcpy(&result, (void *) d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    
    cudaFree((void *) d_result);
    cudaFree(d_random_nums);

    return result;
}

// void random_test() {
//     random_device rd;
//     mt19937_64 eng(rd());
//     uniform_int_distribution<unsigned long long> distr;

//     cout << "Starting Miller-Rabin CUDA test for " << NUM_TEST << " numbers with parameter k = " << K << ". Tests primality with accuracy " << (1 - (1/pow(4, K))) << "." << endl;

//     auto begin = Clock::now();
//     for (int i = 0; i < NUM_TEST; i++) {
//         millerRabinPrimalityTest(distr(eng), K, );
//     }
//     auto end = Clock::now();

//     auto totalDuration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
//     auto avgDuration = ((double) totalDuration) / NUM_TEST;

//     cout << "Total Time: " << totalDuration << " nanoseconds" << endl;
//     cout << "Average Time per iteration: " << avgDuration << " nanoseconds" << endl;
// }

void single_test() {
    random_device rd;
    mt19937_64 eng(rd());
    uniform_int_distribution<unsigned long> distr;

    float *random_nums = new float[K];
    for (int i=0; i<K; i++) {
        random_nums[i] = (float) distr(eng) / (ULONG_MAX);
    }
    int numTest = 10000000;

    cout << "Starting Miller-Rabin CUDA test for " << numTest << " numbers with parameter k = " << K << ". Tests primality with accuracy " << (1 - (1/pow(4, K))) << "." << endl;

    auto begin = Clock::now();
    for (int i = 0; i < numTest; i++) {
        millerRabinPrimalityTest(distr(eng), K, random_nums);
    }
    auto end = Clock::now();

    auto totalDuration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    auto avgDuration = ((double) totalDuration) / numTest;

    cout << "Total Time: " << totalDuration << " nanoseconds" << endl;
    cout << "Average Time per iteration: " << avgDuration << " nanoseconds" << endl;
}

int main(int argc, char const *argv[]) {
    single_test();
    return 0;
}
