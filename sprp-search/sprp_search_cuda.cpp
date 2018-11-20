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

#include "mypair.h"
#include "RandMT.h"
#include "sprp_search_cuda.h"
#include "primesieve/primesieve.hpp"

#define MAX_COMPOSITE 1UL << 32
#define NUM_PAIRS 500000

using namespace std;

typedef chrono::high_resolution_clock Clock;
typedef chrono::time_point<chrono::high_resolution_clock>  ClockTime;


long long executeRound(vector<MyPair> &pairs, vector<unsigned> &primes, unsigned composite_end, float k) {
    cout << "Starting round" << endl;

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

void generatePairs(unsigned a, unsigned count, vector<MyPair> &pairs) {
    RandMT r(time(NULL)); 
    
    for (unsigned i = 0; i < count; i++) {
        pairs.emplace_back(a, r.randomMT(), 9, 4);
    }
}

void single_test() {
    vector<unsigned> primes;
    
    cout << "Starting to generate all primes" << endl;
    primesieve::generate_primes(MAX_COMPOSITE, &primes);
    cout << "Finished generating all primes" << endl;

    vector<MyPair> pairs;
    cout << "Starting to generate " << NUM_PAIRS << " pairs" << endl;
    generatePairs(15, NUM_PAIRS, pairs);
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
