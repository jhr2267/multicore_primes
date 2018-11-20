#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <vector>

#include "mypair.h"
#include "sprp_search_cuda.h"

using namespace thrust;

// Helper function for modular exponentiation.
// Returns a^e (mode n)
__host__ __device__ unsigned long long modexp(unsigned long long a, unsigned long long e, unsigned long long n) {
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
__host__ __device__ bool witnessTest(unsigned long long a, unsigned long long d, unsigned long long n) { 
    unsigned long long x = modexp(a, d, n); 
  
    if (x == 1ULL  || x == n-1) 
       return true;
  
    // Iterate r times (2^r * d = n - 1)
    while (d != n-1) { 
        x = (x * x) % n; 
        d *= 2; 
  
        if (x == 1) {
            return false;
        }
        if (x == n-1) {
            return true;
        }
    }
  
    return false; 
} 
  
// Calls witnessTest for integers "a" and "b" for a known composite number "n"
// Returns true if both integers identify "n" as prime, i.e. "n" is a psuedo-prime for a-SPRP and b-SPRP
// Returns false otherwise
__host__ __device__ bool pairSPRPTest(unsigned long long a, unsigned long long b, unsigned long long d, unsigned long long n) {
    if (witnessTest(a, d, n) && witnessTest(b, d, n)) {
        return true;
    }
    return false;
}

__host__ __device__ bool isComposite(unsigned *primes, unsigned num, MyPair &mypair) {
    if (num == primes[mypair.first_prime_pos]) {
        mypair.first_prime_pos += 1;
        return false;
    } else {
        return true;
    }
}

class findFirstCompositesPairFunctor
{
    private:
        unsigned *primes;
        unsigned composite_end;
    public:
        __host__ __device__ findFirstCompositesPairFunctor(unsigned *d_primes, unsigned c_end) {
            primes = d_primes;
            composite_end = c_end;
        }

        __host__ __device__
        void operator()(MyPair &mypair) {
            bool foundComposite = false;

            for (unsigned j = mypair.first_composite; j <= composite_end; j += 2) {
                if (isComposite(primes, j, mypair)) {
                    unsigned d = j - 1; 
                    while (d % 2 == 0) {
                        d /= 2; 
                    }

                    if (pairSPRPTest(mypair.a, mypair.b, d, j)) {
                        mypair.first_composite = j;
                        foundComposite = true;
                        break;
                    }
                }
            }

            if (!foundComposite) {
                mypair.first_composite = composite_end;
            }
        }
};

void findFirstComposites(std::vector<MyPair> &pairs, std::vector<unsigned> &primes, unsigned composite_end) {
    if (pairs.empty()) {
        return;
    }

    device_vector<MyPair> d_pairs(pairs);
    device_vector<unsigned> d_primes(primes);
    unsigned *d_primes_ptr = thrust::raw_pointer_cast(d_primes.data());

    for_each(thrust::device, d_pairs.begin(), d_pairs.end(), findFirstCompositesPairFunctor(d_primes_ptr, composite_end));

    thrust::copy(d_pairs.begin(), d_pairs.end(), pairs.begin());
}
