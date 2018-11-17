#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <limits.h>
#include <utility>
#include <queue>
#include <vector>
#include <bitset>

#define MAX_COMPOSITE 2 << 10

using namespace std;

class PQCompare {
    public:
    bool operator() (pair<unsigned long long, bool> p1, pair<unsigned long long, bool> p2) {
        return p1.first ? (p2.first ? p1.second > p2.second : true) : (p2.first ? false : p1.second > p2.second);
    }
};

typedef chrono::high_resolution_clock Clock;
typedef priority_queue<pair<unsigned long long, bool>, vector<pair<unsigned long long, bool>>, PQCompare> pairPQ;

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
        d *= 2ULL; 
  
        if (x == 1ULL) {
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

// Given an integer "a", chooses an integer "b" between "lower_bound" and "upper_bound", and runs pairSPRPTest on a list of known composites
void testPairGivenA(unsigned k, unsigned lower_bound, unsigned upper_bound, unsigned long long a, 
    bitset<MAX_COMPOSITE> &composites, unsigned *composites_d, pairPQ pq) {
    
    PQCompare pqcompare;
    
    for (unsigned long long b = lower_bound; b <= upper_bound; b++) {
        bool foundComposite = false;
        for (unsigned i = 9; i < MAX_COMPOSITE; i += 2) {
            if (pairSPRPTest(a, b, composites_d[i], i)) {
                pair<unsigned long long, bool> res = make_pair(i, false);
                if (pq.size() == k && pqcompare(pq.top(), res)) {
                    pq.pop();
                }
                pq.push(res);
                foundComposite = true;
                break;
            }
        }

        if (!foundComposite) {
            pq.push(make_pair(0, true));
        }
    }
}

void single_test() {
    pairPQ pq(PQCompare);
    
    bitset<MAX_COMPOSITE> test;
    sieveOfEratosthenes(test);
    removeEvenComposites(test);
    unsigned * composites_d = calculateD(test);
    for (int i=0; i<100; i++) {
        if (test[i]) {
            cout << i << " " << composites_d[i] << endl;
        }
    }
}

int main(int argc, char const *argv[]) {
    single_test();
    return 0;
}
