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
#include <numeric>
#include <string>
#include <ctime>
#include <memory>
#include <math.h>
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
std::vector<double> WIN;
int sizeOfWindow = 10;

/** Buffer lenght: Number of samples displayed in plot */

const int bufferLength = 500;
/** Setup buffers to accept samples from TSV files */
boost::circular_buffer<double> outerRawBuf(bufferLength);
boost::circular_buffer<double> outerStartBuf(bufferLength);
boost::circular_buffer<double> outerEndBuf(bufferLength);
boost::circular_buffer<double> innerBuf(bufferLength);
boost::circular_buffer<double> innerRawBuf(bufferLength);
boost::circular_buffer<double> snrBuf(bufferLength);
#ifdef DoDeepLearning
boost::circular_buffer<double> removerBuf(bufferLength);
boost::circular_buffer<double> fNNBuf(bufferLength);
boost::circular_buffer<double> l1Buf(bufferLength);
boost::circular_buffer<double> l2Buf(bufferLength);
boost::circular_buffer<double> l3Buf(bufferLength);
#endif
/** Setup LMS buffer results */
boost::circular_buffer<double> lmsBuf(bufferLength);
double outerDelayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> innerDelayLine(innerDelayLineLength);
/** Setup vector to hold SNRs up to interval of grid search */
std::vector<double> snrs;
std::vector<double> outputs;
/** Setup vector to hold max SNRs*/
std::vector<double> maxSnrs;
std::vector<double> minOutputs;
/** Setup constants */
const int subjectNbs = 1;
const int numTrials = 1;
/** Grid Search Interval: */
const int gridInterval = 2000;
#ifdef doOuterDelayLine
int inputNum = outerDelayLineLength;
#else
int inputNum = 1;
#endif

/** Plotting Windows */
#ifdef DoShowPlots
#define WINDOW "data & stat frame"
dynaPlots *plots;
int plotW = 1200;
int plotH = 720;
#endif

/** Setup Neural Network + Parameters */
#ifdef DoDeepLearning
int nNeurons[NLAYERS] = {N22, N21, N20, N19, N18, N17, N16, N15, N14, N13, N12, N11, N10, N9, N8, N7, N6, N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
int numInputs = outerDelayLineLength;
Net *NN = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF");
double wEta = 0;
double bEta = 0;
#endif

//FILTERS
Fir1 *outerFilter[numTrials];
Fir1 *innerFilter[numTrials];
Fir1 *lmsFilter = nullptr;

/** Setup signal types */
double sampleNum, innerRawData, outerRawData;

/** Setup Initial Gains */
double outerGain = 1;
double innerGain = 1;
#ifdef DoDeepLearning
double removerGain = 0;
double feedbackGain = 0;
#endif
/** Setup files to load and save data */
#ifdef DoDeepLearning
fstream nnFile, removerFile, weightFile;
#endif
fstream innerFile;
fstream snrFile;
fstream outerFile;
fstream paramsFile;
fstream lmsFile;
fstream lmsRemoverFile;
fstream laplaceFile;
ifstream eegInfile;

/** Min-Max Values for subjects */
// double minInner[subjectNbs] = {0.063815, 0.018529, 0.04342, -0.058632, 0.022798, 0.014187, 0.031754, 0.038395, 0.024306, 0.025857, 0.036683, 0.023497};
// double minOuter[subjectNbs] = {-0.242428, 0.018594, 0.02451, -0.030434, 0.017505, -0.254623, -0.250294, 0.032478, 0.036081, 0.036793, 0.040581, 0.029097};

// double maxInner[subjectNbs] = {0.065437, 0.020443, 0.04627, -0.045858, 0.025139, 0.03142, 0.034559, 0.020988, 0.023555, 0.02876, 0.037338, 0.025004};
// double maxOuter[subjectNbs] = {-0.237441, 0.0204, 0.026195, -0.016322, 0.019166, -0.252538, -0.249347, 0.03356, 0.037242, 0.037324, 0.041945, 0.031098};
// template <typename T>
std::vector<double> linspace(double start_in, double end_in, double num_in)
{

    std::vector<double> linspaced;
    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}
void print_vector(std::vector<double> vec)
{
    std::cout << "size: " << vec.size() << std::endl;
    for (double d : vec)
        std::cout << d << " ";
    std::cout << std::endl;
}
/** Method to save parameters regarding neural network adjustments */
void saveParam()
{
    paramsFile << "Gains: "
               << "\n"
               << outerGain << "\n"
               << innerGain << "\n"
#ifdef DoDeepLearning
               << removerGain << "\n"
               << feedbackGain << "\n"
               << "Etas: "
               << "\n"
               << wEta << "\n"
               << bEta << "\n"
               << "Network: "
               << "\n"
               << NLAYERS << "\n"
               << N14 << "\n"
               << N13 << "\n"
               << N12 << "\n"
               << N11 << "\n"
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
    paramsFile << "didNoisePreFilter"
               << "\n"
               << maxFilterLength << "\n";
#endif
#ifdef doInnerPreFilter
    paramsFile << "didSignalPreFilter"
               << "\n"
               << maxFilterLength << "\n";
#endif
#ifdef doOuterDelayLine
    paramsFile << "didOuterDelayLine"
               << "\n"
               << outerDelayLineLength << "\n";
#endif
#ifdef doInnerDelay
    paramsFile << "didSignalDelay"
               << "\n"
               << innerDelayLineLength << "\n";
#endif
}
/** Function to delete variables from memory (pointers) */
void freeMemory()
{
#ifdef DoShowPlots
    delete plots;
#endif
#ifdef DoDeepLearning
    delete NN;
#endif
#ifdef doOuterPreFilter
    for (auto &i : outerFilter)
    {
        delete i;
    }
#endif
#ifdef doInnerPreFilter
    for (auto &i : innerFilter)
    {
        delete i;
    }
#endif
    delete lmsFilter;
}
/** Function to automatically close all files */
void handleFiles()
{
    paramsFile.close();
    snrFile.close();
#ifdef DoDeepLearnig
    weightFile.close();
    removerFile.close();
    nnFile.close();
#endif
    innerFile.close();
    outerFile.close();
    lmsFile.close();
    laplaceFile.close();
    lmsRemoverFile.close();
}
/** Setup parameters for grid search */
std::vector<double> outerGainParams = linspace(0.1, 20, 20);
std::vector<double> innerGainParams = linspace(0.1, 20, 20);
std::vector<double> removerGainParams = linspace(0.1, 20, 20);
std::vector<double> feedbackGainParams = linspace(0.1, 1, 20);
std::vector<double> wEtaParams = linspace(0.1, 10, 20);
std::vector<double> wPowParams = linspace(-10, 10, 20);
std::vector<double> bEtaParams = linspace(0.1, 10, 20);
std::vector<double> bPowParams = linspace(-10, 10, 20);
std::vector<std::vector<double>> params = {
    outerGainParams,
    innerGainParams,
    removerGainParams, feedbackGainParams,
    wEtaParams, bEtaParams};
int paramSamples = 250;
double newParam;
double snrFNN;
int main(int argc, const char *argv[])
{

    std::srand(1);
    for (int k = 0; k < subjectNbs; k++) // TODO: CHange back to 0
    {
        // ofstream snrFile;
        int SUBJECT = k;
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
        nnFile.open("./cppData/subject" + sbjct + "/fnn_subject" + sbjct + ".tsv", fstream::out);
        snrFile.open("./cppData/subject" + sbjct + "/snr_subject" + sbjct + ".tsv", fstream::out);
#ifdef DoDeepLearning
        removerFile.open("./cppData/subject" + sbjct + "/remover_subject" + sbjct + ".tsv", fstream::out);
        weightFile.open("./cppData/subject" + sbjct + "/lWeights_closed_subject" + sbjct + ".tsv", fstream::out);
#endif
        innerFile.open("./cppData/subject" + sbjct + "/inner_subject" + sbjct + ".tsv", fstream::out);
        outerFile.open("./cppData/subject" + sbjct + "/outer_subject" + sbjct + ".tsv", fstream::out);
        paramsFile.open("./cppData/subject" + sbjct + "/cppParams_subject" + sbjct + ".tsv", fstream::out);
        lmsFile.open("./cppData/subject" + sbjct + "/lmsOutput_subject" + sbjct + ".tsv", fstream::out);
        lmsRemoverFile.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject" + sbjct + ".tsv", fstream::out);
        laplaceFile.open("./cppData/subject" + sbjct + "/laplace_subject" + sbjct + ".tsv", fstream::out);
        if (!paramsFile)
        {
            cout << "Unable to create files";
            exit(1); // terminate with error
        }
        /** Here Open folder */
        eegInfile.open("./SubjectData/EEG_Subject" + sbjct + ".tsv");

        if (!eegInfile)
        {
            cout << "Unable to open file";
            exit(1); // terminate with error
        }
        //setting up all the filters required
#ifdef doOuterPreFilter
        for (int i = 0; i < numTrials; i++)
        {
            outerFilter[i] = new Fir1("./pyFiles/forOuter.dat");
            outerFilter[i]->reset();
        }
#endif
#ifdef doInnerPreFilter
        for (int i = 0; i < numTrials; i++)
        {
            innerFilter[i] = new Fir1("./pyFiles/forInner.dat");
            innerFilter[i]->reset();
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

        lmsFilter = new Fir1(LMS_COEFF);
        lmsFilter->setLearningRate(LMS_LEARNING_RATE);
        double corrLMS = 0;
        double lmsOutput = 0;

        //setting up the neural network
#ifdef DoDeepLearning
        NN->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
#endif
        int paramCount = 0;
        int inParam = 0;
        double maxSnr;
        while (!eegInfile.eof())
        {

            count += 1;
            // if (paramCount < params.size())
            // {
            //     if (count >= 3000)
            //     {
            //         if (count % 3000 == 0.0)
            //         {
            //             cout << "Param NO: " << paramCount << endl;
            //             inParam = 0;
            //             cout << "Final Vector gains:" << endl;
            //             print_vector(snrs);
            //             int idx = std::distance(snrs.begin(), std::max_element(snrs.begin(), snrs.end()));
            //             snrs = {};
            //             maxSnrs.push_back(maxSnr);
            //             maxSnr = 0;
            //             newParam = params[paramCount][idx];
            //             cout << paramCount << " new param: " << newParam << endl;
            //             plots->set_params(paramCount, SUBJECT, newParam);
            //             // paramSamples = int(gridInterval / params[paramCount].size());
            //             paramCount++;
            //         }
            //     }
            //     if (count > 500 && paramCount < params.size())
            //     {

            //         if (count % 150 == 0.0)
            //         {
            //             if (snrFNN > maxSnr)
            //             {
            //                 maxSnr = snrFNN;
            //             }
            //             snrs.push_back(snrFNN);
            //             inParam++;
            //             newParam = params[paramCount][inParam];
            //             plots->set_params(paramCount, SUBJECT, newParam);
            //         }
            //     }
            // }
            /** Extract Data from TSV files {Inner, Outer} */
            eegInfile >>
                sampleNum >> innerRawData >> outerRawData;
            // GET ALL GAINS:
#ifdef DoDeepLearning
#ifdef DoShowPlots
            innerGain = plots->get_inner_gain(SUBJECT);
            outerGain = plots->get_outer_gain(SUBJECT);
            removerGain = plots->get_remover_gain(SUBJECT);
            feedbackGain = plots->get_feedback_gain(SUBJECT);
#else
            innerGain = 100;
            outerGain = 100;
            removerGain = 10;
            feedbackGain = 1;
#endif
#endif

            /** A) INNER ELECTRODE:  ADJUST & AMPLIFY */
            double innerRaw = 1 * innerGain * (innerRawData); //- minInner[SUBJECT]); //2) FILTERED
#ifdef doInnerPreFilter
            double innerFiltered = innerFilter[0]->filter(innerRaw);
#else
            double innerFiltered = innerRaw;
#endif
            //3) DELAY
#ifdef doInnerDelay
            innerDelayLine.push_back(innerFiltered);
            double inner = innerDelayLine[0];
#else
            double inner = innerFiltered;
#endif
            /** B) OUTER ELECTRODE: ADJUST & AMPLIFY */
            double outerRaw = 1 * outerGain * (outerRawData); // - minOuter[SUBJECT]); //2) FILTERED
#ifdef doOuterPreFilter
            double outer = outerFilter[0]->filter(outerRaw);
#else
            double outer = outerRaw;
#endif
            //3) DELAY LINE
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outerDelayLine[i] = outerDelayLine[i - 1];
            }
            outerDelayLine[0] = outer;
            double *outerDelayed = &outerDelayLine[0];
            // OUTER INPUT TO NETWORK
#ifdef DoDeepLearning
            NN->setInputs(outerDelayed); // Here Input
            NN->propInputs();

            // REMOVER OUTPUT FROM NETWORK
            double removerNN = NN->getOutput(0) * removerGain;
            // cout << removerNN << endl;
            double fNN = (inner - removerNN) * feedbackGain;
            WIN.push_back(fNN);
            // FEEDBACK TO THE NETWORK
            NN->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NN->setBackwardError(fNN);
            NN->propErrorBackward();
#endif
            double sumFNN = std::accumulate(WIN.begin(), WIN.end(), 0);
            double avgFNN = sumFNN / WIN.size();
            double varFNN = 0;
            double sqSumFNN = std::inner_product(WIN.begin(), WIN.end(), WIN.begin(), 0.0);
            double stdFNN = std::sqrt(sqSumFNN / WIN.size() - avgFNN * avgFNN);
            snrFNN = (stdFNN > 0.0) ? avgFNN / stdFNN : 0.0;
            snrFNN = pow(fNN, 2) / pow(outerRaw, 2);
            // snrFNN = fNN / outerRaw;
            /** Network learning */
#ifdef DoDeepLearning
#ifdef DoShowPlots
            wEta = plots->get_wEta(SUBJECT);
            bEta = plots->get_bEta(SUBJECT);
#else
            wEta = 1;
            bEta = 2;
#endif
#endif

#ifdef DoDeepLearning
            NN->setLearningRate(wEta, bEta);
            NN->updateWeights();
            // SAVE WEIGHTS
            for (int i = 0; i < NLAYERS; i++)
            {
                weightFile << NN->getLayerWeightDistance(i) << " ";
            }
            weightFile << NN->getWeightDistance() << "\n";
            NN->snapWeights("cppData", "DNF", SUBJECT);
            double l1 = NN->getLayerWeightDistance(0);
            double l2 = NN->getLayerWeightDistance(1);
            double l3 = NN->getLayerWeightDistance(2);

#endif

            // Do Laplace filter

            double laplace = inner - outer;

            // Do LMS filter
            corrLMS += lmsFilter->filter(outer);
            lmsOutput = inner - corrLMS;

            lmsFilter->lms_update(lmsOutput);

            // SAVE SIGNALS INTO FILES
            laplaceFile << laplace << endl;
            innerFile << inner << endl;
            outerFile << outer << endl;
            snrFile << snrFNN << endl;
#ifdef DoDeepLearning
            removerFile << removerNN << endl;
            nnFile << fNN << endl;
#endif
            lmsFile << lmsOutput << endl;
            lmsRemoverFile << corrLMS << endl;

            // PUT VARIABLES IN BUFFERS
            // 1) MAIN SIGNALS
            outerStartBuf.push_back(outerDelayLine[0]);
            outerEndBuf.push_back(outerDelayLine[outerDelayLineLength - 1]);
            outerRawBuf.push_back(outerRaw);
            innerBuf.push_back(inner);
            innerRawBuf.push_back(innerRaw);
            snrBuf.push_back(snrFNN);
#ifdef DoDeepLearning
            removerBuf.push_back(removerNN);
            fNNBuf.push_back(fNN);
#endif
            // 2) LAYER WEIGHTS
#ifdef DoDeepLearning
            l1Buf.push_back(l1);
            l2Buf.push_back(l2);
            l3Buf.push_back(l3);
#endif

            // 3) LMS outputs
            lmsBuf.push_back(lmsOutput);

            // PUTTING BUFFERS IN VECTORS FOR PLOTS
            // 1) MAIN SIGNALS
            std::vector<double> outerPlot(outerStartBuf.begin(), outerStartBuf.end());
            std::vector<double> outerEndPlot(outerEndBuf.begin(), outerEndBuf.end());
            std::vector<double> outerRawPlot(outerRawBuf.begin(), outerRawBuf.end());
            std::vector<double> innerPlot(innerBuf.begin(), innerBuf.end());
            std::vector<double> innerRawPlot(innerRawBuf.begin(), innerRawBuf.end());
            std::vector<double> snrPlot(snrBuf.begin(), snrBuf.end());
#ifdef DoDeepLearning
            std::vector<double> removerPlot(removerBuf.begin(), removerBuf.end());
            std::vector<double> fNNPlot(fNNBuf.begin(), fNNBuf.end());
            // 2) LAYER WEIGHTS
            std::vector<double> l1Plot(l1Buf.begin(), l1Buf.end());
            std::vector<double> l2Plot(l2Buf.begin(), l2Buf.end());
            std::vector<double> l3Plot(l3Buf.begin(), l3Buf.end());
#endif
            // LMS outputs
            std::vector<double> lmsPlot(lmsBuf.begin(), lmsBuf.end());

            int endTime = time(nullptr);
            int duration = endTime - startTime;

#ifdef DoShowPlots
            frame = cv::Scalar(120, 60, 60);
#ifndef DoDeepLearning
            std::vector<double> removerPlot = {0};
            std::vector<double> fNNPlot = {0};
            std::vector<double> l1Plot = {0};
            std::vector<double> l2Plot = {0};
            std::vector<double> l3Plot = {0};
#endif
            plots->plotMainSignals(outerRawPlot, outerRawPlot, outerRawPlot, innerRawPlot, innerRawPlot, snrPlot, removerPlot, fNNPlot,
                                   l1Plot, l2Plot, l3Plot, lmsPlot, 0);
            plots->plotVariables(SUBJECT);
            // plots->plotSNR(snrPlot);
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
                NN->snapWeights("cppData", "DNF", SUBJECT);
#endif
                handleFiles();
                freeMemory();
                return 0;
            }
        }
        saveParam();
        handleFiles();
        eegInfile.close();
        cout << "Final SNRs" << endl;
        print_vector(maxSnrs);
        // snrFile << snrs;
        cout << "The program has reached the end of the input file" << endl;
    }
    freeMemory();
}
