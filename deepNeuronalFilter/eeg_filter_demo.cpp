#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Iir.h>
#include <Fir1.h>
#include <chrono>
#include <string>
#include <ctime>
#include <memory>
#include <numeric>
#include "cldl/Neuron.h"
#include "cldl/Layer.h"
#include "cldl/Net.h"
#include "parameters.h"
#include "dynamicPlots.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;
constexpr int ESC_key = 27;
int startTime = time(NULL);

//creat circular buffers for plotting
const int bufferLength = 250;
boost::circular_buffer<double> oc_raw_buf(bufferLength);
boost::circular_buffer<double> oc_buf(bufferLength);
boost::circular_buffer<double> oc_end_buf(bufferLength);
boost::circular_buffer<double> ic_buf(bufferLength);
boost::circular_buffer<double> ic_raw_buf(bufferLength);
#ifdef DoDeepLearning
boost::circular_buffer<double> rc_buf(bufferLength);
boost::circular_buffer<double> f_nnc_buf(bufferLength);

boost::circular_buffer<double> l1_c_buf(bufferLength);
boost::circular_buffer<double> l2_c_buf(bufferLength);
boost::circular_buffer<double> l3_c_buf(bufferLength);
#endif
//LMS
boost::circular_buffer<double> lms_c_buf(bufferLength);
boost::circular_buffer<double> lms_o_buf(bufferLength);

//adding delay line for the noise
double outer_closed_delayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> inner_closed_delayLine(innerDelayLineLength);

// CONSTANTS
//const float fs = 250;
const int numTrials = 1;
const int num_subjects = 12;

#ifdef doOuterDelayLine
int num_inputs = outerDelayLineLength;
#else
int num_inputs = 1;
#endif

// PLOT
#ifdef DoShowPlots
#define WINDOW "data & stat frame"
dynaPlots *plots;
int plotW = 1200;
int plotH = 720;
#endif

//NEURAL NETWORK
#ifdef DoDeepLearning
int nNeurons[NLAYERS] = {N10, N9, N8, N7, N6, N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
Net *NNC = new Net(NLAYERS, numNeuronsP, num_inputs, 0, "CLOSED");
double w_eta_closed = 0;
double b_eta_closed = 0;
#endif

//FILTERS
Fir1 *outer_filter[numTrials];
Fir1 *inner_filter[numTrials];
Fir1 *lms_filter_closed = nullptr;

//SIGNALS
double sample_num_closed, inner_closed_raw_data, outer_closed_raw_data, oddball_closed;

// GAINS
double outer_closed_gain = 1;
double inner_closed_gain = 1;
#ifdef DoDeepLearning
double remover_closed_gain = 0;
double feedback_closed_gain = 0;
#endif
// FILES
#ifdef DoDeepLearning
fstream nn_file;
fstream remover_file;
fstream weight_closed_file;
#endif
fstream inner_file;
fstream outer_file;
fstream params_file;
fstream lms_file;
fstream lms_remover_file;
fstream laplace_file;
ifstream closed_infile;
// MIN VALUES
double min_inner_closed[num_subjects] = {0.063815, 0.018529, 0.04342, -0.058632, 0.022798, 0.014187, 0.031754, 0.038395, 0.024306, 0.025857, 0.036683, 0.023497};
double min_outer_closed[num_subjects] = {-0.242428, 0.018594, 0.02451, -0.030434, 0.017505, -0.254623, -0.250294, 0.032478, 0.036081, 0.036793, 0.040581, 0.029097};

double max_inner_closed[num_subjects] = {0.065437, 0.020443, 0.04627, -0.045858, 0.025139, 0.03142, 0.034559, 0.020988, 0.023555, 0.02876, 0.037338, 0.025004};
double max_outer_closed[num_subjects] = {-0.237441, 0.0204, 0.026195, -0.016322, 0.019166, -0.252538, -0.249347, 0.03356, 0.037242, 0.037324, 0.041945, 0.031098};

void saveParam()
{
    params_file << "Gains: "
                << "\n"
                << outer_closed_gain << "\n"
                << inner_closed_gain << "\n"
#ifdef DoDeepLearning
                << remover_closed_gain << "\n"
                << feedback_closed_gain << "\n"
                << "Etas: "
                << "\n"
                << w_eta_closed << "\n"
                << b_eta_closed << "\n"
                << "Network: "
                << "\n"
                << NLAYERS << "\n"
                << N10 << "\n"
                << N9 << "\n"
                << N8 << "\n"
                << N7 << "\n"
                << N6 << "\n"
                << N5 << "\n"
                << N4 << "\n"
                << N3 << "\n"
                << N2 << "\n"
                << N1 << "\n"
                << N0 << "\n"
#endif
                << "LMS"
                << "\n"
                << LMS_COEFF << "\n"
                << LMS_LEARNING_RATE << "\n";

#ifdef doOuterPreFilter
    params_file << "didNoisePreFilter"
                << "\n"
                << maxFilterLength << "\n";
#endif
#ifdef doInnerPreFilter
    params_file << "didSignalPreFilter"
                << "\n"
                << maxFilterLength << "\n";
#endif
#ifdef doOuterDelayLine
    params_file << "didOuterDelayLine"
                << "\n"
                << outerDelayLineLength << "\n";
#endif
#ifdef doInnerDelay
    params_file << "didSignalDelay"
                << "\n"
                << innerDelayLineLength << "\n";
#endif
}

void freeMemory()
{
#ifdef DoShowPlots
    delete plots;
#endif
#ifdef DoDeepLearning
    delete NNC;
#endif
#ifdef doOuterPreFilter
    for (auto &i : outer_filter)
    {
        delete i;
    }
#endif
#ifdef doInnerPreFilter
    for (auto &i : inner_filter)
    {
        delete i;
    }
#endif
    delete lms_filter_closed;
}

void handleFiles()
{
    params_file.close();
#ifdef DoDeepLearnig
    weight_closed_file.close();
    remover_file.close();
    nn_file.close();
#endif
    inner_file.close();
    outer_file.close();
    lms_file.close();
    laplace_file.close();
    lms_remover_file.close();
}

int main(int argc, const char *argv[])
{
    std::srand(1);
    for (int k = 0; k < num_subjects; k++)
    {
        int SUBJECT = k + 1;
        cout << "subject: " << SUBJECT << endl;
        int count = 0;
        //setting up the interactive window and the dynamic plot class
#ifdef DoShowPlots
        auto frame = cv::Mat(cv::Size(plotW, plotH), CV_8UC3);
        cvui::init(WINDOW, 1);
        plots = new dynaPlots(frame, plotW, plotH);
#endif
        //create files for saving the data and parameters
        string sbjct = std::to_string(SUBJECT);
#ifdef DoDeepLearning
        nn_file.open("./cppData/subject" + sbjct + "/fnn_subject" + sbjct + ".tsv", fstream::out);
        remover_file.open("./cppData/subject" + sbjct + "/remover_subject" + sbjct + ".tsv", fstream::out);
        weight_closed_file.open("./cppData/subject" + sbjct + "/lWeights_closed_subject" + sbjct + ".tsv", fstream::out);
#endif
        inner_file.open("./cppData/subject" + sbjct + "/inner_subject" + sbjct + ".tsv", fstream::out);
        outer_file.open("./cppData/subject" + sbjct + "/outer_subject" + sbjct + ".tsv", fstream::out);
        params_file.open("./cppData/subject" + sbjct + "/cppParams_subject" + sbjct + ".tsv", fstream::out);
        lms_file.open("./cppData/subject" + sbjct + "/lmsOutput_subject" + sbjct + ".tsv", fstream::out);
        lms_remover_file.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject" + sbjct + ".tsv", fstream::out);
        laplace_file.open("./cppData/subject" + sbjct + "/laplace_subject" + sbjct + ".tsv", fstream::out);

        if (!params_file)
        {
            cout << "Unable to create files";
            exit(1); // terminate with error
        }
        // Here Open folder
        closed_infile.open("./SubjectData/EEG_Subject" + sbjct + ".tsv");

        if (!closed_infile)
        {
            cout << "Unable to open file";
            exit(1); // terminate with error
        }

        //setting up all the filters required
#ifdef doOuterPreFilter
        for (int i = 0; i < numTrials; i++)
        {
            outer_filter[i] = new Fir1("./pyFiles/forOuter.dat");
            outer_filter[i]->reset();
        }
#endif
#ifdef doInnerPreFilter
        for (int i = 0; i < numTrials; i++)
        {
            inner_filter[i] = new Fir1("./pyFiles/forInner.dat");
            inner_filter[i]->reset();
        }
#endif
#ifdef doOuterPreFilter
        int waitOutFilterDelay = maxFilterLength;
#else
#ifdef doInnerPreFilter
        int waitOutFilterDelay = maxFilterLength;
#else
        int waitOutFilterDelay = 1;
#endif
#endif

        lms_filter_closed = new Fir1(LMS_COEFF);
        lms_filter_closed->setLearningRate(LMS_LEARNING_RATE);

        double corrLMS_closed = 0;
        double lms_output_closed = 0;
        //    const float fs = 250;

        //setting up the neural networks
#ifdef DoDeepLearning
        NNC->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
#endif

        while (!closed_infile.eof())
        {
            //for (int loop=0; loop < 5000; loop++){
            count += 1;
            //get the data from .tsv files:
            closed_infile >> sample_num_closed >> inner_closed_raw_data >> outer_closed_raw_data >> oddball_closed;
            // GET ALL GAINS:
#ifdef DoDeepLearning
#ifdef DoShowPlots
            inner_closed_gain = plots->get_inner_gain(0);
            outer_closed_gain = plots->get_outer_gain(0);
            remover_closed_gain = plots->get_remover_gain(0);
            feedback_closed_gain = plots->get_feedback_gain(0);
#else
            inner_closed_gain = 100;
            outer_closed_gain = 100;
            remover_closed_gain = 10;
            feedback_closed_gain = 1;
#endif
#endif

            //A) INNER ELECTRODE:
            //1) ADJUST & AMPLIFY
            double inner_closed_raw = 1 * inner_closed_gain * (inner_closed_raw_data - min_inner_closed[SUBJECT]); // - min_inner_closed[SUBJECT])/(max_inner_closed[SUBJECT] - min_inner_closed[SUBJECT]);
                                                                                                                   //2) FILTERED
#ifdef doInnerPreFilter
            double inner_closed_filtered = inner_filter[0]->filter(inner_closed_raw);
#else
            double inner_closed_filtered = inner_closed_raw;
#endif
            //3) DELAY
#ifdef doInnerDelay
            inner_closed_delayLine.push_back(inner_closed_filtered);
            double inner_closed = inner_closed_delayLine[0];
#else
            double inner_closed = inner_closed_filtered;
#endif
            //B) OUTER ELECTRODE:
            //1) ADJUST & AMPLIFY
            double outer_closed_raw = 1 * outer_closed_gain * (outer_closed_raw_data - min_outer_closed[SUBJECT]); // - min_outer_closed[SUBJECT])/(max_outer_closed[SUBJECT] - min_outer_closed[SUBJECT]);
                                                                                                                   //2) FILTERED
#ifdef doOuterPreFilter
            double outer_closed = outer_filter[0]->filter(outer_closed_raw);
#else
            double outer_closed = outer_closed_raw;
#endif
            //3) DELAY LINE
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outer_closed_delayLine[i] = outer_closed_delayLine[i - 1];
            }
            outer_closed_delayLine[0] = outer_closed;
            double *outer_closed_delayed = &outer_closed_delayLine[0];

            //cout << outer_closed << outer_closed_delayLine[0] << outer_closed_delayLine[1] << outer_closed_delayLine[2] << endl;

            // OUTER INPUT TO NETWORK
#ifdef DoDeepLearning
            NNC->setInputs(outer_closed_delayed); // Here Input
            NNC->propInputs();

            // REMOVER OUTPUT FROM NETWORK
            double remover_closed = NNC->getOutput(0) * remover_closed_gain;
            double f_nn_closed = (inner_closed - remover_closed) * feedback_closed_gain;

            // FEEDBACK TO THE NETWORK
            NNC->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NNC->setBackwardError(f_nn_closed);
            NNC->propErrorBackward();
#endif

            // LEARN
#ifdef DoDeepLearning
#ifdef DoShowPlots
            w_eta_closed = plots->get_wEta(0);
            b_eta_closed = plots->get_bEta(0);
#else
            w_eta_closed = 1;
            b_eta_closed = 2;
#endif
#endif

#ifdef DoDeepLearning
            NNC->setLearningRate(w_eta_closed, b_eta_closed);
            //if (count > maxFilterLength+outerDelayLineLength){
            NNC->updateWeights();
            //}
            // SAVE WEIGHTS
            for (int i = 0; i < NLAYERS; i++)
            {
                weight_closed_file << NNC->getLayerWeightDistance(i) << " ";
            }
            weight_closed_file << NNC->getWeightDistance() << "\n";
            NNC->snapWeights("cppData", "closed", SUBJECT);
            double l1_c = NNC->getLayerWeightDistance(0);
            double l2_c = NNC->getLayerWeightDistance(1);
            double l3_c = NNC->getLayerWeightDistance(2);
#endif

            // Do Laplace filter

            double laplace_closed = inner_closed - outer_closed;

            // Do LMS filter
            corrLMS_closed += lms_filter_closed->filter(outer_closed);
            lms_output_closed = inner_closed - corrLMS_closed;

            lms_filter_closed->lms_update(lms_output_closed);

            // SAVE SIGNALS INTO FILES
            laplace_file << laplace_closed << endl;
            inner_file << inner_closed << endl;
            outer_file << outer_closed << endl;
#ifdef DoDeepLearning
            remover_file << remover_closed << endl;
            nn_file << f_nn_closed << endl;
#endif
            lms_file << lms_output_closed << endl;
            lms_remover_file << corrLMS_closed << endl;

            // PUT VARIABLES IN BUFFERS
            // 1) MAIN SIGNALS
            //      CLOSED EYES
            oc_buf.push_back(outer_closed_delayLine[0]);
            oc_end_buf.push_back(outer_closed_delayLine[outerDelayLineLength - 1]);
            oc_raw_buf.push_back(outer_closed_raw);
            ic_buf.push_back(inner_closed);
            ic_raw_buf.push_back(inner_closed_raw);
#ifdef DoDeepLearning
            rc_buf.push_back(remover_closed);
            f_nnc_buf.push_back(f_nn_closed);
#endif
            // 2) LAYER WEIGHTS
#ifdef DoDeepLearning
            l1_c_buf.push_back(l1_c);
            l2_c_buf.push_back(l2_c);
            l3_c_buf.push_back(l3_c);
#endif

            // 3) LMS outputs
            lms_c_buf.push_back(lms_output_closed);

            // PUTTING BUFFERS IN VECTORS FOR PLOTS
            // 1) MAIN SIGNALS
            //      CLOSED EYES
            std::vector<double> oc_plot(oc_buf.begin(), oc_buf.end());
            std::vector<double> oc_end_plot(oc_end_buf.begin(), oc_end_buf.end());
            std::vector<double> oc_raw_plot(oc_raw_buf.begin(), oc_raw_buf.end());
            std::vector<double> ic_plot(ic_buf.begin(), ic_buf.end());
            std::vector<double> ic_raw_plot(ic_raw_buf.begin(), ic_raw_buf.end());
#ifdef DoDeepLearning
            std::vector<double> rc_plot(rc_buf.begin(), rc_buf.end());
            std::vector<double> f_nnc_plot(f_nnc_buf.begin(), f_nnc_buf.end());
            // 2) LAYER WEIGHTS
            std::vector<double> l1_c_plot(l1_c_buf.begin(), l1_c_buf.end());
            std::vector<double> l2_c_plot(l2_c_buf.begin(), l2_c_buf.end());
            std::vector<double> l3_c_plot(l3_c_buf.begin(), l3_c_buf.end());
#endif
            // LMS outputs
            std::vector<double> lms_c_plot(lms_c_buf.begin(), lms_c_buf.end());

            int endTime = time(nullptr);
            int duration = endTime - startTime;

#ifdef DoShowPlots
            frame = cv::Scalar(60, 60, 60);
#ifndef DoDeepLearning
            std::vector<double> f_nnc_plot = {0};
            std::vector<double> rc_plot = {0};
            std::vector<double> l1_c_plot = {0};
            std::vector<double> l2_c_plot = {0};
            std::vector<double> l3_c_plot = {0};
#endif
            plots->plotMainSignals(oc_raw_plot, oc_plot, oc_end_plot,
                                   ic_raw_plot, ic_plot, rc_plot, f_nnc_plot,
                                   l1_c_plot, l2_c_plot, l3_c_plot, lms_c_plot, 0);
            plots->plotVariables(0);
            plots->plotVariables(1);
            plots->plotTitle(count, duration);
            cvui::update();
            cv::imshow(WINDOW, frame);
#endif

            /**
 * If the Esc button is pressed on the interactive window the final SNRs are printed on the console and
 * all SNRs and parameters are saved to a file. Also all pointers are deleted to free dynamically allocated memory.
 * Then the files are closed and program returns with 0.
 */
            if (cv::waitKey(20) == ESC_key)
            {
                saveParam();
#ifdef DoDeepLearning
                NNC->snapWeights("cppData", "closed", SUBJECT);
#endif
                handleFiles();
                freeMemory();
                return 0;
            }
        }
        saveParam();
        handleFiles();
        closed_infile.close();
        cout << "The program has reached the end of the input file" << endl;
    }
    freeMemory();
}
