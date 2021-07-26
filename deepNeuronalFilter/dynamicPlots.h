#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Iir.h>
#include <Fir1.h>
#include <string>
#include <numeric>
#include <memory>
#include <opencv2/opencv.hpp>
#include "cvui.h"

using namespace std;

namespace cv
{
    class Mat;
}

class dynaPlots
{
public:
    dynaPlots(cv::Mat &_learningFrame, int _plotW, int _plotH);
    ~dynaPlots();

    void plotSNR(std::vector<double> snr);

    void plotMainSignals(std::vector<double> outer_raw, std::vector<double> outer, std::vector<double> outer_end,
                         std::vector<double> inner_raw, std::vector<double> inner, std::vector<double> snr,
                         std::vector<double> remover, std::vector<double> fnn,
                         std::vector<double> l1_plot, const std::vector<double> &l2_plot, const std::vector<double> &l3_plot,
                         std::vector<double> lms_output,
                         int _positionOPEN);
    void plotVariables(int closed_or_open);

    void plotTitle(int count, int duration);

    inline double get_wEta(int closed_or_open) { return (wEta[closed_or_open] * pow(10, wEtaPower[closed_or_open])); }
    inline double get_bEta(int closed_or_open) { return (bEta[closed_or_open] * pow(10, bEtaPower[closed_or_open])); }
    inline double get_outer_gain(int closed_or_open) { return outer_gain[closed_or_open]; }
    inline double get_inner_gain(int closed_or_open) { return inner_gain[closed_or_open]; }
    inline double get_remover_gain(int closed_or_open) { return remover_gain[closed_or_open]; }
    inline double get_feedback_gain(int closed_or_open) { return feedback_gain[closed_or_open]; }
    inline void set_wEta(int closed_or_open, double wEtaNew, double wEtaPowNew) { wEta[closed_or_open] = wEtaNew * pow(10, wEtaPowNew); }
    inline void set_bEta(int closed_or_open, double bEtaNew, double bEtaPowNew) { bEta[closed_or_open] = bEtaNew * pow(10, bEtaPowNew); }
    inline void set_outer_gain(int closed_or_open, double outerGainNew) { outer_gain[closed_or_open] = outerGainNew; }
    inline void set_inner_gain(int closed_or_open, double innerGainNew) { inner_gain[closed_or_open] = innerGainNew; }
    inline void set_remover_gain(int closed_or_open, double removerGainNew) { remover_gain[closed_or_open] = removerGainNew; }
    inline void set_feedback_gain(int closed_or_open, double feedbackGainNew) { feedback_gain[closed_or_open] = feedbackGainNew; }
    inline void set_params(int idx, int closed_or_open, double newVal)
    {
        switch (idx)
        {
        case 0:
            set_outer_gain(closed_or_open, newVal);
            break;
        case 1:
            set_inner_gain(closed_or_open, newVal);
            break;
        case 2:
            set_remover_gain(closed_or_open, newVal);
            break;
        case 3:
            set_feedback_gain(closed_or_open, newVal);
            break;
        case 4:
            set_wEta(closed_or_open, newVal, 0.0);
            break;
        case 5:
            set_bEta(closed_or_open, newVal, 0.0);
            break;
        default:
            break;
        }
    }

private:
    cv::Mat frame;

    // noise type          {relax , blink, sudoku}
    // Relax parameters Locked in.
    double outer_gain[3] = {
        1.1,
        0.1,
        0.1};
    double inner_gain[3] = {
        18.9526,
        15.8105,
        18.9};
    double remover_gain[3] = {8.5, 8.4, 8.4};
    double feedback_gain[3] = {0.1, 1, 0.1};
    double wEta[3] = {0.1, 1.66316, 1.58};
    double wEtaPower[3] = {0.5, 0.5, 0.5};
    double bEta[3] = {1.2, 0.621053, 1.2};
    double bEtaPower[3] = {-0.5, -0.5, -0.5};

    double gainStart = 0.0;
    double gainEnd = 20.0;

    int topOffset = 30;
    int graphDX = 360;
    int graphDY = 110;
    int graphY = 0;
    int gapY = 30;
    int gapX = 15;
    int barY = 60;
    int lineEnter = 15;
    int barDX = 200;
    int bar_p = 1;
    int titleY = 2;

    int plotH;
    int plotW;
};
