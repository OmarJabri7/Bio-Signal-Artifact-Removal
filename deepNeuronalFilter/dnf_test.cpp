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

ifstream eegInfilePure;
ifstream eegInfileNoisy;

int subjectsNb = 3;

std::vector<double> minErrors;

double sampleNum, sampleNumNoisy, outerDataPure, innerDataPure, outerDataNoisy, innerDataNoisy;
double newParam;
int paramCount, inParam = 0;
double errorNN;
double outerGain, innerGain, removerGain, fnnGain, w, b;
double outerPureDelayLine[outerDelayLineLength] = {0.0};
double outerNoisyDelayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> innerPureDelayLine(innerDelayLineLength);
boost::circular_buffer<double> innerNoisyDelayLine(innerDelayLineLength);
int numInputs = outerDelayLineLength;
int nNeurons[NLAYERS] = {N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
Net *NNP = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF Pure");
Net *NNN = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF Noisy");
const int numTrials = 2;
Fir1 *outerFilter[numTrials];
Fir1 *innerFilter[numTrials];
Fir1 *lmsFilterPure = nullptr;
Fir1 *lmsFilterNoisy = nullptr;

void closeFiles()
{
    paramsFile.close();

    eegInfilePure.close();
    eegInfileNoisy.close();

    weightFile.close();
    removerFile.close();
    nnFile.close();
    innerFile.close();
    outerFile.close();
    lmsFile.close();
    lmsRemoverFile.close();
}
int main(int argc, const char *argv[])
{
    outerGain = 5;
    innerGain = 1;
    removerGain = 2.5;
    fnnGain = 1;
    w = 0.5;
    b = 0;
    cout << "Outer gain: " << outerGain << endl;
    cout << "Inner gain: " << innerGain << endl;
    cout << "Remover gain: " << removerGain << endl;
    cout << "Feedback gain: " << fnnGain << endl;
    cout << "Weight ETA: " << w << endl;
    cout << "Bias ETA: " << b << endl;

    for (int s = 0; s < subjectsNb; s++)
    {
        cout << NLAYERS << endl;
        double startTime = time(NULL);
        int SUBJECT = s;
        int count = 0;
        cout << "subject: " << SUBJECT << endl;
        string sbjct = std::to_string(SUBJECT);

        paramsFile.open("./cppData/subject" + sbjct + "/cppParams_subject_Alpha" + sbjct + ".tsv", fstream::out);

        nnFile.open("./cppData/subject" + sbjct + "/fnn_subject_Alpha" + sbjct + ".tsv", fstream::out);
        removerFile.open("./cppData/subject" + sbjct + "/remover_subject_Alpha" + sbjct + ".tsv", fstream::out);
        weightFile.open("./cppData/subject" + sbjct + "/lWeights_closed_subject_Alpha" + sbjct + ".tsv", fstream::out);
        innerFile.open("./cppData/subject" + sbjct + "/inner_subject_Alpha" + sbjct + ".tsv", fstream::out);
        outerFile.open("./cppData/subject" + sbjct + "/outer_subject_Alpha" + sbjct + ".tsv", fstream::out);
        lmsFile.open("./cppData/subject" + sbjct + "/lmsOutput_subject_Alpha" + sbjct + ".tsv", fstream::out);
        lmsRemoverFile.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject_Alpha" + sbjct + ".tsv", fstream::out);
        eegInfilePure.open("./SubjectData/Alpha/Pure/EEG_Subject" + sbjct + ".tsv");

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
        lmsFilterPure = new Fir1(LMS_COEFF);
        lmsFilterPure->setLearningRate(LMS_LEARNING_RATE);
        lmsFilterNoisy = new Fir1(LMS_COEFF);
        lmsFilterNoisy->setLearningRate(LMS_LEARNING_RATE);
        double corrLMSPure = 0;
        double lmsOutputPure = 0;
        double corrLMSNoisy = 0;
        double lmsOutputNoisy = 0;
        NNP->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        NNN->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        double minError;
        int lineNb = 0;
        std::string line;
        while (!eegInfilePure.eof())
        {
            std::getline(eegInfilePure, line);
            lineNb++;
        }
        cout << lineNb << endl;
        progresscpp::ProgressBar progressBar(lineNb, 70);
        eegInfilePure.close();
        eegInfilePure.open("./SubjectData/Alpha/Pure/EEG_Subject" + sbjct + ".tsv");
        eegInfileNoisy.open("./SubjectData/Alpha/Noisy/EEG_Subject" + sbjct + ".tsv");

        while (!eegInfilePure.eof() && !eegInfileNoisy.eof())
        {
            ++progressBar;
            progressBar.display();
            count += 1;
            // eegInfilePure >>
            // sampleNum >> innerDataPure >> outerDataPure;
            eegInfileNoisy >>
                sampleNumNoisy >> innerDataNoisy >> outerDataNoisy;
            // outerDataPure *= outerGain;
            // innerDataPure *= innerGain;
            outerDataNoisy *= outerGain;
            innerDataNoisy *= innerGain;
            // double innerFilteredPure = innerFilter[0]->filter(innerDataPure);
            double innerFilteredNoisy = innerFilter[1]->filter(innerDataNoisy);
            // innerPureDelayLine.push_back(innerDataPure);
            // double innerPure = innerPureDelayLine[0];
            innerNoisyDelayLine.push_back(innerDataNoisy);
            double innerNoisy = innerNoisyDelayLine[0];

            // double outerPure = outerFilter[0]->filter(outerDataPure);
            double outerNoisy = outerFilter[1]->filter(outerDataNoisy);
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                // outerPureDelayLine[i] = outerPureDelayLine[i - 1];
                outerNoisyDelayLine[i] = outerNoisyDelayLine[i - 1];
            }
            // outerPureDelayLine[0] = outerDataPure;
            outerNoisyDelayLine[0] = outerDataNoisy;
            // double *outerDelayedPure = &outerPureDelayLine[0];
            double *outerDelayedNoisy = &outerNoisyDelayLine[0];

            // NNP->setInputs(outerDelayedPure);  // Here Input
            NNN->setInputs(outerDelayedNoisy); // Here Input
            // NNP->propInputs();
            NNN->propInputs();
            // double removerPure = NNP->getOutput(0) * removerGain;
            double removerNoisy = NNN->getOutput(0) * removerGain;
            // double fNNPure = (innerPure - removerPure) * fnnGain;
            double fNNNoisy = (innerNoisy - removerNoisy) * fnnGain;
            removerFile
                << "-1"
                << " " << removerNoisy << endl;
            cout << "-1" << removerNoisy << endl;
            nnFile << "-1"
                   << " " << fNNNoisy << endl;
            // NNP->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NNN->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            // NNP->setBackwardError(fNNPure);
            NNN->setBackwardError(fNNNoisy);
            // NNP->propErrorBackward();
            NNN->propErrorBackward();

            // w /= count;
            // NNP->setLearningRate(w, b);
            NNN->setLearningRate(w, b);

            // NNP->updateWeights();
            NNN->updateWeights();

            // NNP->snapWeights("cppData", "Pure", SUBJECT);
            NNN->snapWeights("cppData", "Noisy", SUBJECT);

            // corrLMSPure += lmsFilterPure->filter(outerDataPure);
            corrLMSNoisy += lmsFilterNoisy->filter(outerDataNoisy);

            // lmsOutputPure = innerPure - corrLMSPure;
            lmsOutputNoisy = innerNoisy - corrLMSNoisy;

            // lmsFilterPure->lms_update(lmsOutputPure);
            lmsFilterNoisy->lms_update(lmsOutputNoisy);

            lmsFile << "-1"
                    << " " << lmsOutputNoisy << endl;
            lmsRemoverFile << "-1"
                           << " " << corrLMSNoisy << endl;
            innerFile << "-1"
                      << " " << innerNoisy << endl;
            outerFile << "-1"
                      << " " << outerNoisy << endl;
        }
        progressBar.done();
        paramsFile << NLAYERS << endl;
        paramsFile << outerDelayLineLength << endl;
        paramsFile << innerDelayLineLength << endl;
        paramsFile << outerGain << endl;
        paramsFile << innerGain << endl;
        paramsFile << removerGain << endl;
        paramsFile << fnnGain << endl;
        paramsFile << w << endl;
        paramsFile << b << endl;
        double endTime = time(nullptr);
        double duration = endTime - startTime;
        cout << "Duration: " << double(duration / 60.0) << " mins" << endl;
        closeFiles();
        cout << "The program has reached the end of the input file" << endl;
    }
}