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

void generatePairs(unsigned count, vector<MyPair> &pairs) {
    RandMT r(time(NULL)); 
    
    for (unsigned i = 0; i < count; i++) {
        pairs.emplace_back(r.randomMT(), r.randomMT(), 9, 4);
    }
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

    executeFourRounds(pairs, primes, 0.3);
}

int main(int argc, char const *argv[]) {
    single_test();
    return 0;
}
