//
// Created by sama on 25/06/19.
//
#ifndef EEGFILTER_PARAMETERS_H
#define EEGFILTER_PARAMETERS_H

#define LMS_COEFF (int)(250)
#define LMS_LEARNING_RATE 0.00001

#define DoShowPlots

#define maxFilterLength 250

// NOISE:
#define doOuterPreFilter
#define doOuterDelayLine
#define outerDelayLineLength 1 // CHange this

// SIGNAL:
#define doInnerPreFilter
#define doInnerDelay
#define innerDelayLineLength 1 // Change this

//NN specifications
#define DoDeepLearning
#define NLAYERS 5

#define N32 101
#define N31 96
#define N30 92
#define N29 87
#define N28 82

#define N27 101
#define N26 96
#define N25 92
#define N24 87
#define N23 82

#define N22 101
#define N21 96
#define N20 92
#define N19 87
#define N18 82
#define N17 75
#define N16 62
#define N15 58
#define N14 47
#define N13 42
#define N12 101
#define N11 59
#define N10 37
#define N9 23
#define N8 19
#define N7 17
#define N6 13
#define N5 11
#define N4 7
#define N3 5
#define N2 3
#define N1 2
#define N0 1

#endif //EEGFILTER_PARAMETERS_H