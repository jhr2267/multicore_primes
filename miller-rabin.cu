#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <limits>
#include <cuda.h>
typedef std::chrono::high_resolution_clock Clock;

#define NUM_TEST 10000000
#define NUM_BLOCKS 1
#define NUM_THREADS 256
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
__device__ bool witnessTest(unsigned long long d, unsigned long long n, float random_num) { 
    // Pick a random number in [2..n-2] 
    // Corner cases make sure that n > 4 
    unsigned long long a = random_num * (n-4) + 2;
    unsigned long long x = modexp(a, d, n); 
  
    if (x == 1ULL  || x == n-1) 
       return true; 
  
    // Iterate r times (2^r * d = n - 1)
    while (d != n-1) { 
        x = (x * x) % n; 
        d *= 2ULL; 
  
        if (x == 1ULL) {
            return false;
        }
        if (x == n-1) {
            return true;
        }
    }
  
    // Return composite 
    return false; 
} 
  
// See: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
// Returns true if k-probably prime (k is a parameter that determines accuracy)
// Returns false if composite
__global__ void millerRabinPrimalityTest(unsigned long long *nums, unsigned len, bool *isPrime, unsigned long long k, float *random_nums) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    
    int n = nums[idx];
    if (n == 4ULL) {
        isPrime[idx] = false;
        return;
    }
    if (n <= 3ULL) {
        isPrime[idx] = true;
        return;
    }
  
    // Find r such that n = 2^d * r + 1 for some r >= 1 
    unsigned long long d = n - 1; 
    while (d % 2 == 0ULL) {
        d /= 2ULL; 
    }
  
    // Witness loop to repeat k times
    for (unsigned long long i = 0; i < k; i++) {
        if (!witnessTest(d, n, random_nums[k])){
            isPrime[idx] = false;
            return;
        }
    }

    isPrime[idx] = true;
}

void random_test() {
    random_device rd;
    mt19937_64 eng(rd());
    uniform_int_distribution<unsigned long> distr;

    float *random_nums = new float[K];
    for (int i=0; i<K; i++) {
        random_nums[i] = (float) distr(eng) / (ULONG_MAX);
    }

    cout << "Starting Miller-Rabin CUDA test for " << NUM_TEST << " numbers with parameter k = " << K << ". Tests primality with accuracy " << (1 - (1/pow(4, K))) << "." << endl;

    auto begin = Clock::now();
    vector<unsigned long long> test;
    for (int i = 0; i < NUM_TEST; i++) {
        test.push_back(distr(eng));
    }

    unsigned long long *d_nums;
    bool *d_isPrime;
    float *d_random_nums;
    bool *isPrime = new bool[NUM_TEST];
    
    cudaMalloc((void **) &d_random_nums, K * sizeof(float));
    cudaMalloc((void **) &d_isPrime, NUM_TEST * sizeof(bool));
    cudaMalloc((void **) &d_nums, NUM_TEST * sizeof(unsigned long long));
    

    cudaMemcpy((void *) d_nums, test.data(), NUM_TEST * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_random_nums, random_nums, K * sizeof(float), cudaMemcpyHostToDevice);

    millerRabinPrimalityTest<<<(NUM_TEST + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(d_nums, test.size(), d_isPrime, K, d_random_nums);

    cudaMemcpy(isPrime, d_isPrime, NUM_TEST * sizeof(bool), cudaMemcpyDeviceToHost);

    auto end = Clock::now();

    auto totalDuration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    auto avgDuration = ((double) totalDuration) / NUM_TEST;

    cout << "Total Time: " << totalDuration << " nanoseconds" << endl;
    cout << "Average Time per iteration: " << avgDuration << " nanoseconds" << endl;

    cudaFree(d_isPrime);
    cudaFree(d_nums);
    cudaFree(d_random_nums);
    delete[] isPrime;
}

void single_test() {
    random_device rd;
    mt19937_64 eng(rd());
    uniform_int_distribution<unsigned long> distr;

    float *random_nums = new float[K];
    for (int i=0; i<K; i++) {
        random_nums[i] = (float) distr(eng) / (ULONG_MAX);
    }

    int numTest = 10;
    cout << "Starting Miller-Rabin CUDA test for " << numTest << " numbers with parameter k = " << K << ". Tests primality with accuracy " << (1 - (1/pow(4, K))) << "." << endl;

    auto begin = Clock::now();
    vector<unsigned long long> test;
    for (int i = 0; i < numTest; i++) {
        test.push_back(distr(eng));
    }

    unsigned long long *d_nums;
    bool *d_isPrime;
    float *d_random_nums;
    bool *isPrime = new bool[numTest];
    
    cudaMalloc((void **) &d_random_nums, K * sizeof(float));
    cudaMalloc((void **) &d_isPrime, numTest * sizeof(bool));
    cudaMalloc((void **) &d_nums, numTest * sizeof(unsigned long long));
    

    cudaMemcpy((void *) d_nums, test.data(), numTest * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_random_nums, random_nums, K * sizeof(float), cudaMemcpyHostToDevice);

    millerRabinPrimalityTest<<<(numTest + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(d_nums, test.size(), d_isPrime, K, d_random_nums);

    cudaMemcpy(isPrime, d_isPrime, numTest * sizeof(bool), cudaMemcpyDeviceToHost);

    auto end = Clock::now();

    for (int i=0; i<numTest; i++) {
        cout << test[i] << " is prime: " << isPrime[i] << endl;
    }

    auto totalDuration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    auto avgDuration = ((double) totalDuration) / numTest;

    cout << "Total Time: " << totalDuration << " nanoseconds" << endl;
    cout << "Average Time per iteration: " << avgDuration << " nanoseconds" << endl;

    cudaFree(d_isPrime);
    cudaFree(d_nums);
    cudaFree(d_random_nums);
    delete[] isPrime;
}

int main(int argc, char const *argv[]) {
    random_test();
    return 0;
}
