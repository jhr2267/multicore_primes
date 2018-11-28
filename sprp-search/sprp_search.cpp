#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <limits.h>
#include <utility>
#include <queue>
#include <vector>
#include <algorithm>
#include <bitset>

#include "RandMT.h"
#include "mypair.h"
#include "primesieve/primesieve.hpp"

#define MAX_COMPOSITE 1UL << 32
#define NUM_PAIRS 500000
#define FIRST_NUMBER_LIMIT 10000

using namespace std;

typedef chrono::high_resolution_clock Clock;
typedef chrono::time_point<chrono::high_resolution_clock>  ClockTime;

// Helper function for modular exponentiation.
// Returns a^e (mode n)
unsigned long long modexp(unsigned long long a, unsigned long long e, unsigned long long n) {
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
bool witnessTest(unsigned long long a, unsigned long long d, unsigned long long n) { 
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
bool pairSPRPTest(unsigned long long a, unsigned long long b, unsigned long long d, unsigned long long n) {
    if (witnessTest(a, d, n) && witnessTest(b, d, n)) {
        return true;
    }
    return false;
}

void generatePairs(unsigned count, vector<MyPair> &pairs) {
    RandMT r(time(NULL)); 
    
    for (unsigned i = 0; i < count; i++) {
        pairs.emplace_back(r.randomMT() % FIRST_NUMBER_LIMIT, r.randomMT(), 9, 4);
    }
}


bool isComposite(vector<unsigned> &primes, unsigned num, MyPair &pair) {
    if (num == primes[pair.first_prime_pos]) {
        pair.first_prime_pos += 1;
        return false;
    } else {
        return true;
    }
}

void findFirstComposites(vector<MyPair> &pairs, vector<unsigned> &primes, unsigned composite_end) {
    if (pairs.empty()) {
        return;
    }

    unsigned size = pairs.size();
    for (unsigned i = 0; i < size; i++) {
        bool foundComposite = false;
    
        for (unsigned j = pairs[i].first_composite; j <= composite_end; j += 2) {
            if (isComposite(primes, j, pairs[i])) {
                unsigned d = j - 1; 
                while (d % 2 == 0) {
                    d /= 2;
                }

                if (pairSPRPTest(pairs[i].a, pairs[i].b, d, j)) {
                    pairs[i].first_composite = j;
                    foundComposite = true;
                    break;
                }
            }
        }

        if (!foundComposite) {
            pairs[i].first_composite = composite_end;
        }
    }
}

// Needed so no compiler error
MyPair::MyPair() {}

long long executeRound(vector<MyPair> &pairs, vector<unsigned> &primes, unsigned composite_end, float k) {
    cout << "Starting sequential round" << endl;

    ClockTime begin = Clock::now();
    
    findFirstComposites(pairs, primes, composite_end);
    sort(pairs.begin(), pairs.end(), MyPair::compare());
    
    ClockTime end = Clock::now();
    long long duration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    cout << "Round Time: " << duration << " nanoseconds\n" << endl;
    
    cout << "Size before truncating: " << pairs.size() << endl;
    size_t resize_len = (size_t) (k * pairs.size()) < 10 ? 10 : (k * pairs.size());
    pairs.resize(resize_len);
    cout << "Size after truncating with factor " << k << ": " << pairs.size() << endl;

    cout << "Top 10 pairs (a, b, first_composite): " << endl;
    unsigned len =  pairs.size() < 10U ? pairs.size() : 10U;
    for (unsigned i = 0; i < len; i++) {
        cout << pairs[i].a << " " << pairs[i].b << " " << pairs[i].first_composite << endl;
    }

    cout << endl;
    return duration;
}

void single_test() {
    vector<unsigned> primes;
    
    cout << "Starting to generate all primes" << endl;
    primesieve::generate_primes(MAX_COMPOSITE, &primes);
    cout << "Finished generating all primes" << endl;

    vector<MyPair> pairs;
    cout << "Starting to generate " << NUM_PAIRS << " pairs" << endl;
    generatePairs(NUM_PAIRS, pairs);
    cout << "Finished generating " << NUM_PAIRS << " pairs" << endl;

    long long totalDuration = 0;
    totalDuration += executeRound(pairs, primes, 1000005, 0.3);
    totalDuration += executeRound(pairs, primes, 100000005, 0.3);
    totalDuration += executeRound(pairs, primes, 1000000005, 0.3);
    totalDuration += executeRound(pairs, primes, 4294967295, 0.3);

    cout << "Total running time was: " << totalDuration << " nanoseconds" << endl;
}

int main(int argc, char const *argv[]) {
    single_test();
    return 0;
}
