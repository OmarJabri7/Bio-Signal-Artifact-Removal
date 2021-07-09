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

/** Buffer lenght: Number of samples displayed in plot */

const int bufferLength = 250;
/** Setup buffers to accept samples from TSV files */
boost::circular_buffer<double> outerRawBuf(bufferLength);
boost::circular_buffer<double> outerStartBuf(bufferLength);
boost::circular_buffer<double> outerEndBuf(bufferLength);
boost::circular_buffer<double> innerBuf(bufferLength);
boost::circular_buffer<double> innerRawBuf(bufferLength);
#ifdef DoDeepLearning
boost::circular_buffer<double> removerBuf(bufferLength);
boost::circular_buffer<double> fNNBuf(bufferLength);
boost::circular_buffer<double> l1Buf(bufferLength);
boost::circular_buffer<double> l2Buf(bufferLength);
boost::circular_buffer<double> l3Buf(bufferLength);
#endif
/** Setup LMS buffer results */
boost::circular_buffer<double> lmsBuf(bufferLength);

/** Setup constants */
const int subjectNbs = 12;
const int trialNbs = 1;

/** Plotting Windows */
#ifdef DoShowPlots
#define WINDOW "data & stat frame"
dynaPlots *plots;
int plotW = 1200;
int plotH = 680;
#endif

/** Setup Neural Network + Parameters */
#ifdef DoDeepLearning
int nNeurons[NLAYERS] = {N10, N9, N8, N7, N6, N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
int numInputs = 1;
Net *NN = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF");
double wEta = 0;
double bEta = 0;
#endif

Fir1 *lmsFilter = nullptr;
/** Setup signal types */
double sampleNum, innerRawData, outerRawData, oddBall;

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
fstream outerFile;
fstream paramsFile;
fstream lmsFile;
fstream lmsRemoverFile;
fstream laplaceFile;
ifstream eegInfile;

/** Min-Max Values for subjects */
double minInner[subjectNbs] = {0.063815, 0.018529, 0.04342, -0.058632, 0.022798, 0.014187, 0.031754, 0.038395, 0.024306, 0.025857, 0.036683, 0.023497};
double minOuter[subjectNbs] = {-0.242428, 0.018594, 0.02451, -0.030434, 0.017505, -0.254623, -0.250294, 0.032478, 0.036081, 0.036793, 0.040581, 0.029097};

double maxInner[subjectNbs] = {0.065437, 0.020443, 0.04627, -0.045858, 0.025139, 0.03142, 0.034559, 0.020988, 0.023555, 0.02876, 0.037338, 0.025004};
double maxOuter[subjectNbs] = {-0.237441, 0.0204, 0.026195, -0.016322, 0.019166, -0.252538, -0.249347, 0.03356, 0.037242, 0.037324, 0.041945, 0.031098};

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
    delete lmsFilter;
}
/** Functin to automatically close all files */
void handleFiles()
{
    paramsFile.close();
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

int main(int argc, const char *argv[])
{
    std::srand(1);
    for (int k = 0; k < subjectNbs; k++)
    {
        int SUBJECT = k + 1;
        cout << "subject: " << SUBJECT << endl;
        int count = 0;
        /** Setting up the interactive window and the dynamic plot class */
#ifdef DoShowPlots
        auto frame = cv::Mat(cv::Size(plotW, plotH), CV_8UC3);
        cvui::init(WINDOW, 1);
        plots = new dynaPlots(frame, plotW, plotH);
#endif
        /** Create files and save data */
        string sbjct = std::to_string(SUBJECT);
#ifdef DoDeepLearning
        nnFile.open("./cppData/subject" + sbjct + "/fnn_subject" + sbjct + ".tsv", fstream::out);
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
        lmsFilter = new Fir1(LMS_COEFF);
        lmsFilter->setLearningRate(LMS_LEARNING_RATE);
        double corrLMS = 0;
        double lmsOutput = 0;
#ifdef DoDeepLearning
        NN->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
#endif
        while (!eegInfile.eof())
        {
            count += 1;
            /** Extract Data from TSV files {Inner, Outer} */
            eegInfile >> sampleNum >> innerRawData >> outerRawData >> oddBall;
#ifdef DoDeepLearning
#ifdef DoShowPlots
            innerGain = plots->get_inner_gain(0);
            outerGain = plots->get_outer_gain(0);
            removerGain = plots->get_remover_gain(0);
            feedbackGain = plots->get_feedback_gain(0);
#else
            innerGain = 100;
            outerGain = 100;
            removerGain = 10;
            feedbackGain = 1;
#endif
#endif
            /** A) INNER ELECTRODE:  ADJUST & AMPLIFY */
            double innerRaw = 1 * innerGain * (innerRawData - minInner[SUBJECT]);
            /** B) OUTER ELECTRODE: ADJUST & AMPLIFY */
            double outerRaw = 1 * outerGain * (outerRawData - minOuter[SUBJECT]);
#ifdef DoDeepLearning
            NN->setInputs(&outerRaw); // Here Input
            NN->propInputs();
            // REMOVER OUTPUT FROM NETWORK
            double removerNN = NN->getOutput(0) * removerGain;
            double fNN = (innerRaw - removerNN) * feedbackGain;
            // FEEDBACK TO THE NETWORK
            NN->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NN->setBackwardError(fNN);
            NN->propErrorBackward();
#endif
/** Neural Network Learning */
#ifdef DoDeepLearning
#ifdef DoShowPlots
            wEta = plots->get_wEta(0);
            bEta = plots->get_bEta(0);
#else
            wEta = 1;
            bEta = 2;
#endif
#endif
#ifdef DoDeepLearning
            NN->setLearningRate(wEta, bEta);
            NN->updateWeights();
            /** SAVE WEIGHTS */
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
            /** Do Laplace and LMS Filtering */
            double laplace = innerRaw - outerRaw;

            // Do LMS filter
            corrLMS += lmsFilter->filter(outerRaw);
            lmsOutput = innerRaw - corrLMS;

            lmsFilter->lms_update(lmsOutput);

            // SAVE SIGNALS INTO FILES
            laplaceFile << laplace << endl;
            innerFile << innerRaw << endl;
            outerFile << outerRaw << endl;
#ifdef DoDeepLearning
            removerFile << removerNN << endl;
            nnFile << fNN << endl;
#endif
            lmsFile << lmsOutput << endl;
            lmsRemoverFile << corrLMS << endl;
            /** PUT VARIABLES IN BUFFERS
                 1) MAIN SIGNALS */
            outerRawBuf.push_back(outerRaw);
            innerRawBuf.push_back(innerRaw);
#ifdef DoDeepLearning
            removerBuf.push_back(removerNN);
            fNNBuf.push_back(fNN);
#endif
            /** 2) LAYER WEIGHTS */
#ifdef DoDeepLearning
            l1Buf.push_back(l1);
            l2Buf.push_back(l2);
            l3Buf.push_back(l3);
#endif
            /** 3) LMS outputs */
            lmsBuf.push_back(lmsOutput);
            /** PUTTING BUFFERS IN VECTORS FOR PLOTS
             1) MAIN SIGNALS */
            std::vector<double> outerRawPlot(outerRawBuf.begin(), outerRawBuf.end());
            std::vector<double> innerPlot(innerBuf.begin(), innerBuf.end());
            std::vector<double> innerRawPlot(innerRawBuf.begin(), innerRawBuf.end());
#ifdef DoDeepLearning
            std::vector<double> removerPlot(removerBuf.begin(), removerBuf.end());
            std::vector<double> fNNPlot(fNNBuf.begin(), fNNBuf.end());
            /** 2) LAYER WEIGHTS */
            std::vector<double> l1Plot(l1Buf.begin(), l1Buf.end());
            std::vector<double> l2Plot(l2Buf.begin(), l2Buf.end());
            std::vector<double> l3Plot(l3Buf.begin(), l3Buf.end());
#endif
            /** 3) LMS outputs */
            std::vector<double> lmsPlot(lmsBuf.begin(), lmsBuf.end());
            int endTime = time(nullptr);
            int duration = endTime - startTime;
#ifdef DoShowPlots
            frame = cv::Scalar(255, 255, 255);
#endif
#ifndef DoDeepLearning
            std::vector<double> fNNPlot = {0};
            std::vector<double> removerPlot = {0};
            std::vector<double> l1Plot = {0};
            std::vector<double> l2Plot = {0};
            std::vector<double> l3Plot = {0};
#endif
            std::vector<double> tmp1 = {0};
            std::vector<double> tmp2 = {0};
            plots->plotMainSignals(tmp1, outerRawPlot, tmp2,
                                   tmp1, innerRawPlot, removerPlot, fNNPlot,
                                   l1Plot, l2Plot, l3Plot, lmsPlot, 0);
            plots->plotVariables(0);
            plots->plotTitle(count, duration);
            cvui::update();
            cv::imshow(WINDOW, frame);
            // #endif
            /*   *If the Esc button is pressed on the interactive window the final SNRs are printed on the console and
                    *all SNRs and parameters are saved to a file.Also all pointers are deleted to free dynamically allocated memory.
                        *Then the files are closed and program returns with 0. *
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
        cout << "The program has reached the end of the input file" << endl;
    }
    freeMemory();
}
