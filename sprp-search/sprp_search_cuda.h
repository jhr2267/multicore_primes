#ifndef _SPRP_SEARCH_CUDA_H
#define _SPRP_SEARCH_CUDA_H

#include "mypair.h"
#include <vector>

void executeFourRounds(std::vector<MyPair> &pairs, std::vector<unsigned> &primes, float k);

#endif