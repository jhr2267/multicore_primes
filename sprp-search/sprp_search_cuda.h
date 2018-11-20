#ifndef _SPRP_SEARCH_CUDA_H
#define _SPRP_SEARCH_CUDA_H

#include "mypair.h"
#include <vector>

void findFirstComposites(std::vector<MyPair> &pairs, std::vector<unsigned> &primes, unsigned composite_end);

#endif