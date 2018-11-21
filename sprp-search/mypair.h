#ifndef _MYPAIR_H
#define _MYPAIR_H

class MyPair {
    public:
    unsigned a, b, first_composite, first_prime_pos;

    MyPair();

    MyPair(unsigned a, unsigned b, unsigned first_composite, unsigned first_prime_pos);

    struct compare {
        bool operator() (const MyPair &p1, const MyPair &p2) {
            return p1.first_composite > p2.first_composite;
        }
    };
};

#endif