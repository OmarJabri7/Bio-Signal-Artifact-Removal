//
//  eegGeneration.cpp
//
//
//  Created by Omar Jabri on 03/06/2021.
//

#include "eegGeneration.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

int main()
{
    int A;
    int T;   /* s */
    int f = 15; /* Hz */
    int fs;
    int phase = 0;
    int N = 100;
    vector<int> x(N);
    iota(begin(x), end(x), 0);
    auto eeg = sin(x);
}
