#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <limits>
typedef std::chrono::high_resolution_clock Clock;

#define NUM_TEST 100000
#define K 100

using namespace std;
  
// Utility function to do modular exponentiation. 
// It returns (x^y) % p 

unsigned long long modexp(unsigned long long a, unsigned long long e, unsigned long long n)
{
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
// Returns false if composite or true if 
bool witnessTest(unsigned long long d, unsigned long long n) 
{ 
    // Pick a random number in [2..n-2] 
    // Corner cases make sure that n > 4 
    unsigned long long a = 2 + rand() % (n - 4); 
    unsigned long long x = modexp(a, d, n); 
  
    if (x == 1  || x == n-1) 
       return true; 
  
    // Iterate r times (2^r * d = n - 1)
    while (d != n-1) { 
        x = (x * x) % n; 
        d *= 2; 
  
        if (x == 1)      return false; 
        if (x == n-1)    return true; 
    } 
  
    // Return composite 
    return false; 
} 
  
// See: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
// Returns true if k-probably prime (k is a parameter that determines accuracy)
// Returns false if composite
bool millerRabinPrimalityTest(unsigned long long n, unsigned long long k) 
{
    if (n == 4) return false; 
    if (n <= 3) return true; 
  
    // Find r such that n = 2^d * r + 1 for some r >= 1 
    unsigned long long d = n - 1; 
    while (d % 2 == 0) {
        d /= 2; 
    }
  
    // Witness loop to repeat k times
    for (unsigned long long i = 0; i < k; i++) {
        if (!witnessTest(d, n)) 
            return false; 
    }
  
    return true; 
}

int main(int argc, char const *argv[])
{
    random_device rd;
    mt19937_64 eng(rd());
    uniform_int_distribution<unsigned long long> distr;

    cout << "Starting Miller-Rabin test for " << NUM_TEST << " numbers with parameter k = " << K << ". Tests primality with accuracy " << (1 - (1/pow(4, K))) << "." << endl;

    auto begin = Clock::now();
    for (int i = 0; i < NUM_TEST; i++) {
        millerRabinPrimalityTest(distr(eng), K);
    }
    auto end = Clock::now();

    auto totalDuration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    auto avgDuration = ((double) totalDuration) / NUM_TEST;

    cout << "Total Time: " << totalDuration << " nanoseconds" << endl;
    cout << "Average Time per iteration: " << avgDuration << " nanoseconds" << endl;
    return 0;
}
