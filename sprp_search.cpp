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
#include "primesieve/primesieve.hpp"

#define MAX_COMPOSITE 1UL << 32
#define NUM_PAIRS 5000

using namespace std;

struct Pair {
    unsigned a, b, first_composite;

    Pair() {}

    Pair(unsigned a, unsigned b, unsigned first_composite) {
        this->a = a;
        this->b = b;
        this->first_composite = first_composite;
    }

    struct compare {
        bool operator() (const Pair &p1, const Pair &p2) {
            return p1.first_composite > p2.first_composite;
        }
    };
};

typedef chrono::high_resolution_clock Clock;

// Sieve of Eratosthenese test
// See: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Pseudocode
// High bits are composite and low bits are prime
void sieveOfEratosthenes(bitset<MAX_COMPOSITE> &composites) {
    unsigned N = (unsigned) sqrt(MAX_COMPOSITE);
    for (unsigned i = 2; i < N; i++) {
        if (!composites[i]) {
            for (unsigned j = i*i; j < MAX_COMPOSITE; j += i) {
                composites[j] = true;
            }
        }
    }
}

// Helper function to remove even composites in the bit array
void removeEvenComposites(bitset<MAX_COMPOSITE> &composites) {
    for (unsigned i = 0; i < MAX_COMPOSITE; i += 2) {
        composites[i] = false;
    }
}

unsigned * calculateD(bitset<MAX_COMPOSITE> &composites) {
    unsigned *composites_d = new unsigned[MAX_COMPOSITE];

    for (unsigned i = 9; i < MAX_COMPOSITE; i += 2) {
        if (composites[i]) {
            unsigned d = i - 1; 
            while (d % 2 == 0) {
                d /= 2; 
            }
            composites_d[i] = d;
        }
    }

    return composites_d;
}

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
    // cout << a << " " << b << " " << d << " " << n << endl;
    if (witnessTest(a, d, n) && witnessTest(b, d, n)) {
        return true;
    }
    return false;
}

void generatePairs(unsigned a, unsigned count, vector<Pair> &pairs) {
    RandMT r(time(NULL)); 
    
    for (unsigned i = 0; i < count; i++) {
        pairs.emplace_back(a, r.randomMT(), 9);
    }
}


pair<unsigned, bool> isComposite(vector<unsigned> &primes, unsigned num, unsigned pos) {
    if (num == primes[pos]) {
        return make_pair(pos+1, false);
    } else if (num < primes[pos]) {
        return make_pair(pos, true);
    }
}

void findFirstComposites(vector<Pair> &pairs, vector<unsigned> &primes, unsigned composite_start, unsigned composite_end) {
    if (pairs.empty()) {
        return;
    }

    unsigned size = pairs.size();
    for (unsigned i = 0; i < size; i++) {
        if (i % (size/10) == 0) {
            cout << ((float) i / size) * 100 << "% done" << endl;
        }

        bool foundComposite = false;
        vector<unsigned>::iterator it = lower_bound(primes.begin(), primes.end(), pairs[i].first_composite);
        unsigned pos = it - primes.begin();
        pair<unsigned, bool> res = make_pair(pos, false);

        for (unsigned j = pairs[i].first_composite; j <= composite_end; j += 2) {
            res = isComposite(primes, j, res.first);
            if (res.second) {
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


void executeRound(vector<Pair> &pairs, vector<unsigned> &primes, unsigned composite_start, unsigned composite_end, float k) {
    cout << "Starting round" << endl;

    findFirstComposites(pairs, primes, composite_start, composite_end);

    sort(pairs.begin(), pairs.end(), Pair::compare());

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
}

void single_test() {
    vector<unsigned> primes;
    
    cout << "Starting to generate all primes" << endl;
    primesieve::generate_primes(MAX_COMPOSITE, &primes);
    cout << "Finished generating all primes" << endl;

    vector<Pair> pairs;

    cout << "Starting to generate pairs" << endl;
    generatePairs(15, NUM_PAIRS, pairs);
    cout << "Finished generating pairs" << endl;

    executeRound(pairs, primes, 9, 1000005, 0.3);
    executeRound(pairs, primes, 1000005, 100000005, 0.3);
    executeRound(pairs, primes, 100000005, 1000000005, 0.3);
    executeRound(pairs, primes, 1000000005, 4294967295, 1);
}

int main(int argc, char const *argv[]) {
    single_test();
    return 0;
}
