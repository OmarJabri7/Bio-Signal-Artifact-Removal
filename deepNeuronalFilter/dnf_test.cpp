#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <Iir.h>
#include <Fir1.h>
#include <boost/circular_buffer.hpp>
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
#include "tqdm.h"
#include "ProgressBar.h"

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

/** Setup parameters for grid search */
std::vector<double> removerGainParams = linspace(0.1, 20, 20);
std::vector<double> feedbackGainParams = linspace(0.1, 1, 20);
std::vector<double> wEtaParams = linspace(0.1, 10, 20);
std::vector<double> bEtaParams = linspace(0.1, 10, 20);
std::vector<std::vector<double>> params = {
    removerGainParams, feedbackGainParams,
    wEtaParams, bEtaParams};
std::vector<double> errors;

fstream paramsFile;

fstream nnFile, removerFile, weightFile;
fstream innerFile;
fstream outerFile;
fstream lmsFile;
fstream lmsRemoverFile;
fstream laplaceFile;
fstream errorFile;

ifstream signalToUse;
ifstream eegInfileNoisy;

int subjectsNb = 2;

std::vector<double> minErrors;

double sampleNum, sampleNumNoisy, outerDataNoisy, innerDataNoisy;
double newParam;
int paramCount, inParam = 0;
double errorNN;
double outerGain, innerGain, removerGain, fnnGain, w, b;
double outerNoisyDelayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> innerNoisyDelayLine(innerDelayLineLength);
int numInputs = outerDelayLineLength;
int nNeurons[NLAYERS] = {N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
Net *NNN = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF Noisy");
const int numTrials = 2;
Fir1 *outerFilter[numTrials];
Fir1 *innerFilter[numTrials];
Fir1 *lmsFilterNoisy = nullptr;

void closeFiles()
{
    paramsFile.close();

    eegInfileNoisy.close();
    signalToUse.close();
    weightFile.close();
    removerFile.close();
    nnFile.close();
    innerFile.close();
    outerFile.close();
    lmsFile.close();
    lmsRemoverFile.close();
    laplaceFile.close();
    errorFile.close();
}
int main(int argc, const char *argv[])
{
    signalToUse.open("signal.txt");
    string signals;
    signalToUse >> signals;
    cout << signals << ":" << endl;
    // if (signals == "Alpha")
    // {
    const float outerGain[] = {20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0};
    const float innerGain[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const float removerGain[] = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5};
    const float fnnGain[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const float w[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    const float b[] = {0, 0, 0, 0, 0, 0, 0};
    // }
    // else if (signals == "Delta")
    // {
    //     cout << "HIBS USHS" << endl;
    //     const float outerGain[] = {20.0, 20.0, 20.0, 20.0, 20.0};
    //     const float innerGain[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    //     const float removerGain[] = {2.5, 2.5, 2.5, 2.5, 2.5};
    //     const float fnnGain[] = {1, 1, 1, 1, 1};
    //     const float w[] = {0.5, 0.5, 0.0001, 0.5, 0.5};
    //     const float b[] = {0, 0, 0, 0, 0};
    // }
    for (int s = 0; s < subjectsNb; s++)
    {
        cout << NLAYERS << endl;
        cout << outerGain[s] << endl;
        cout << innerGain[s] << endl;
        cout << removerGain[s] << endl;
        cout << fnnGain[s] << endl;
        cout << w[s] << endl;
        cout << b[s] << endl;
        double startTime = time(NULL);
        int SUBJECT = s;
        int count = 0;
        cout << "subject: " << SUBJECT << endl;
        string sbjct = std::to_string(SUBJECT);

        paramsFile.open("./cppData/subject" + sbjct + "/cppParams_subject_" + signals + sbjct + ".tsv", fstream::out);

        nnFile.open("./cppData/subject" + sbjct + "/fnn_subject_" + signals + sbjct + ".tsv", fstream::out);
        removerFile.open("./cppData/subject" + sbjct + "/remover_subject_" + signals + sbjct + ".tsv", fstream::out);
        weightFile.open("./cppData/subject" + sbjct + "/lWeights_closed_subject_" + signals + sbjct + ".tsv", fstream::out);
        innerFile.open("./cppData/subject" + sbjct + "/inner_subject_" + signals + sbjct + ".tsv", fstream::out);
        outerFile.open("./cppData/subject" + sbjct + "/outer_subject_" + signals + sbjct + ".tsv", fstream::out);
        lmsFile.open("./cppData/subject" + sbjct + "/lmsOutput_subject_" + signals + sbjct + ".tsv", fstream::out);
        lmsRemoverFile.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject_" + signals + sbjct + ".tsv", fstream::out);
        laplaceFile.open("./cppData/subject" + sbjct + "/laplace_subject_" + signals + sbjct + ".tsv", fstream::out);
        eegInfileNoisy.open("./SubjectData/" + signals + "/Noisy/EEG_Subject" + sbjct + ".tsv");
        errorFile.open("./cppData/subject" + sbjct + "/error_subject_" + signals + sbjct + ".tsv", fstream::out);

        for (int i = 0; i < numTrials; i++)
        {
            outerFilter[i] = new Fir1("./pyFiles/forOuter.dat");
            outerFilter[i]->reset();
        }
        for (int i = 0; i < numTrials; i++)
        {
            innerFilter[i] = new Fir1("./pyFiles/forInner.dat");
            innerFilter[i]->reset();
        }
        lmsFilterNoisy = new Fir1(LMS_COEFF);
        lmsFilterNoisy->setLearningRate(LMS_LEARNING_RATE);
        double corrLMSNoisy = 0;
        double lmsOutputNoisy = 0;
        NNN->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        double minError;
        int lineNb = 0;
        std::string line;
        while (!eegInfileNoisy.eof())
        {
            std::getline(eegInfileNoisy, line);
            lineNb++;
        }
        cout << lineNb << endl;
        progresscpp::ProgressBar progressBar(lineNb, 70);
        eegInfileNoisy.close();

        eegInfileNoisy.open("./SubjectData/" + signals + "/Noisy/EEG_Subject" + sbjct + ".tsv");

        while (!eegInfileNoisy.eof())
        {
            ++progressBar;
            progressBar.display();
            count += 1;
            eegInfileNoisy >>
                sampleNumNoisy >> innerDataNoisy >> outerDataNoisy;
            outerDataNoisy *= outerGain[s];
            innerDataNoisy *= innerGain[s];
            double innerFilteredNoisy = innerFilter[1]->filter(innerDataNoisy);
            innerNoisyDelayLine.push_back(innerDataNoisy);
            double innerNoisy = innerNoisyDelayLine[0];

            double outerNoisy = outerFilter[1]->filter(outerDataNoisy);
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outerNoisyDelayLine[i] = outerNoisyDelayLine[i - 1];
            }
            outerNoisyDelayLine[0] = outerDataNoisy;
            double *outerDelayedNoisy = &outerNoisyDelayLine[0];

            NNN->setInputs(outerDelayedNoisy); // Here Input
            NNN->propInputs();
            double removerNoisy = NNN->getOutput(0) * removerGain[s];
            double fNNNoisy = (innerNoisy - removerNoisy) * fnnGain[s];
            removerFile
                << "-1"
                << " " << removerNoisy << endl;
            nnFile << "-1"
                   << " " << fNNNoisy << endl;
            NNN->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NNN->setBackwardError(fNNNoisy);
            NNN->propErrorBackward();

            NNN->setLearningRate(w[s], b[s]);

            NNN->updateWeights();

            NNN->snapWeights("cppData", "Noisy", SUBJECT);
            double laplace = innerDataNoisy - outerDataNoisy;

            corrLMSNoisy += lmsFilterNoisy->filter(outerDataNoisy);

            lmsOutputNoisy = innerDataNoisy - corrLMSNoisy;

            lmsFilterNoisy->lms_update(lmsOutputNoisy);
            laplaceFile << "-1"
                        << " " << laplace << endl;
            lmsFile << "-1"
                    << " " << lmsOutputNoisy << endl;
            lmsRemoverFile << "-1"
                           << " " << corrLMSNoisy << endl;
            innerFile << "-1"
                      << " " << innerNoisy << endl;
            outerFile << "-1"
                      << " " << outerNoisy << endl;
            errorFile << fNNNoisy << " " << lmsOutputNoisy << " " << laplace << endl;
        }
        progressBar.done();
        paramsFile << NLAYERS << endl;
        paramsFile << outerDelayLineLength << endl;
        paramsFile << innerDelayLineLength << endl;
        paramsFile << outerGain[s] << endl;
        paramsFile << innerGain[s] << endl;
        paramsFile << removerGain[s] << endl;
        paramsFile << fnnGain[s] << endl;
        paramsFile << w[s] << endl;
        paramsFile << b[s] << endl;
        double endTime = time(nullptr);
        double duration = endTime - startTime;
        cout << "Duration: " << double(duration / 60.0) << " mins" << endl;
        closeFiles();
        cout << "The program has reached the end of the input file" << endl;
    }
}