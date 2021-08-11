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

fstream nnFileAlpha, removerFileAlpha, weightFileAlpha;
fstream innerFileAlpha;
fstream outerFileAlpha;
fstream lmsFileAlpha;
fstream lmsRemoverFileAlpha;
fstream laplaceFileAlpha;
ifstream eegInfileAlpha;

fstream nnFileDelta, removerFileDelta, weightFileDelta;
fstream innerFileDelta;
fstream outerFileDelta;
fstream lmsFileDelta;
fstream lmsRemoverFileDelta;
fstream laplaceFileDelta;
ifstream eegInfileDelta;

int subjectsNb = 3;

std::vector<double> minErrors;

double sampleNum, outerDataAlpha, innerDataAlpha, outerDataDelta, innerDataDelta;
double newParam;
int paramCount, inParam = 0;
double errorNN;
double outerGainAlpha, innerGainAlpha, removerGainAlpha, fnnGainAlpha, wAlpha, bAlpha;
double outerGainDelta, innerGainDelta, removerGainDelta, fnnGainDelta, wDelta, bDelta;
double outerDelayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> innerAlphaDelayLine(innerDelayLineLength);
boost::circular_buffer<double> innerDeltaDelayLine(innerDelayLineLength);
int numInputs = outerDelayLineLength;
// int numInputs = 1;
int nNeurons[NLAYERS] = {
    N12, N11, N10, N9, N8, N7, N6, N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
Net *NNA = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF Alpha");
Net *NND = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF Delta");
const int numTrials = 1;
Fir1 *outerFilter[numTrials];
Fir1 *innerFilter[numTrials];
Fir1 *lmsFilterAlpha = nullptr;
Fir1 *lmsFilterDelta = nullptr;
void closeFiles()
{
    paramsFile.close();

    eegInfileAlpha.close();
    weightFileAlpha.close();
    removerFileAlpha.close();
    nnFileAlpha.close();
    innerFileAlpha.close();
    outerFileAlpha.close();
    lmsFileAlpha.close();
    lmsRemoverFileAlpha.close();

    eegInfileDelta.close();
    weightFileDelta.close();
    removerFileDelta.close();
    nnFileDelta.close();
    innerFileDelta.close();
    outerFileDelta.close();
    lmsFileDelta.close();
    lmsRemoverFileDelta.close();
}
int main(int argc, const char *argv[])
{
    outerGainAlpha = 1;
    innerGainAlpha = 1;
    removerGainAlpha = 10;
    fnnGainAlpha = 1;
    wAlpha = 0.1;
    bAlpha = 0;
    outerGainDelta = 20;
    innerGainDelta = 0.5;
    removerGainDelta = 1;
    fnnGainDelta = 0.9;
    wDelta = 0.01;
    bDelta = 0;
    cout << "Alpha parameters: " << endl;
    cout << "Outer gain: " << outerGainAlpha << endl;
    cout << "Inner gain: " << innerGainAlpha << endl;
    cout << "Remover gain: " << removerGainAlpha << endl;
    cout << "Feedback gain: " << fnnGainAlpha << endl;
    cout << "Weight ETA: " << wAlpha << endl;
    cout << "Bias ETA: " << bAlpha << endl;
    cout << "Delta parameters: " << endl;
    cout << "Outer gain: " << outerGainDelta << endl;
    cout << "Inner gain: " << innerGainDelta << endl;
    cout << "Remover gain: " << removerGainDelta << endl;
    cout << "Feedback gain: " << fnnGainDelta << endl;
    cout << "Weight ETA: " << wDelta << endl;
    cout << "Bias ETA: " << bDelta << endl;

    for (int s = 0; s < subjectsNb; s++)
    {
        cout << NLAYERS << endl;
        double startTime = time(NULL);
        int SUBJECT = s;
        int count = 0;
        cout << "subject: " << SUBJECT << endl;
        string sbjct = std::to_string(SUBJECT);

        paramsFile.open("./cppData/subject" + sbjct + "/cppParams_subject" + sbjct + ".tsv", fstream::out);

        nnFileAlpha.open("./cppData/subject" + sbjct + "/fnn_subject_Alpha" + sbjct + ".tsv", fstream::out);
        removerFileAlpha.open("./cppData/subject" + sbjct + "/remover_subject_Alpha" + sbjct + ".tsv", fstream::out);
        weightFileAlpha.open("./cppData/subject" + sbjct + "/lWeights_closed_subject_Alpha" + sbjct + ".tsv", fstream::out);
        innerFileAlpha.open("./cppData/subject" + sbjct + "/inner_subject_Alpha" + sbjct + ".tsv", fstream::out);
        outerFileAlpha.open("./cppData/subject" + sbjct + "/outer_subject_Alpha" + sbjct + ".tsv", fstream::out);
        lmsFileAlpha.open("./cppData/subject" + sbjct + "/lmsOutput_subject_Alpha" + sbjct + ".tsv", fstream::out);
        lmsRemoverFileAlpha.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject_Alpha" + sbjct + ".tsv", fstream::out);
        eegInfileAlpha.open("./SubjectData/Alpha/EEG_Subject" + sbjct + ".tsv");
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
        lmsFilterAlpha = new Fir1(LMS_COEFF);
        lmsFilterAlpha->setLearningRate(LMS_LEARNING_RATE);
        double corrLMSAlpha = 0;
        double lmsOutputAlpha = 0;
        NNA->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        double minError;
        int lineNb = 0;
        std::string line;
        while (!eegInfileAlpha.eof())
        {
            std::getline(eegInfileAlpha, line);
            lineNb++;
        }
        cout << lineNb << endl;
        progresscpp::ProgressBar progressBar(lineNb, 70);
        eegInfileAlpha.close();
        eegInfileAlpha.open("./SubjectData/Alpha/EEG_Subject" + sbjct + ".tsv");
        cout << "Alpha waves start:" << endl;
        while (!eegInfileAlpha.eof())
        {
            ++progressBar;
            progressBar.display();
            count += 1;
            eegInfileAlpha >>
                sampleNum >> innerDataAlpha >> outerDataAlpha;
            outerDataAlpha *= outerGainAlpha;
            innerDataAlpha *= innerGainAlpha;
            double innerFilteredAlpha = innerFilter[0]->filter(innerDataAlpha);
            innerAlphaDelayLine.push_back(innerFilteredAlpha);
            double innerAlpha = innerAlphaDelayLine[0];
            double outer = outerFilter[0]->filter(outerDataAlpha);
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outerDelayLine[i] = outerDelayLine[i - 1];
            }
            outerDelayLine[0] = outer;
            double *outerDelayed = &outerDelayLine[0];
            NNA->setInputs(outerDelayed); // Here Input
            NNA->propInputs();
            double removerAlpha = NNA->getOutput(0) * removerGainAlpha;
            double fNNAlpha = (innerAlpha - removerAlpha) * fnnGainAlpha;
            // double snrFNN = pow(fNN, 2) / pow(outerData, 2);
            removerFileAlpha
                << removerAlpha << endl;
            nnFileAlpha << fNNAlpha << endl;
            NNA->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NNA->setBackwardError(fNNAlpha);
            NNA->propErrorBackward();
            // w /= count;
            NNA->setLearningRate(wAlpha, bAlpha);
            NNA->updateWeights();
            NNA->snapWeights("cppData", "Alpha", SUBJECT);
            corrLMSAlpha += lmsFilterAlpha->filter(outer);
            lmsOutputAlpha = innerAlpha - corrLMSAlpha;
            lmsFilterAlpha->lms_update(lmsOutputAlpha);
            lmsFileAlpha << lmsOutputAlpha << endl;
            lmsRemoverFileAlpha << corrLMSAlpha << endl;
            innerFileAlpha << innerAlpha << endl;
            outerFileAlpha << outer << endl;
        }
        nnFileDelta.open("./cppData/subject" + sbjct + "/fnn_subject_Delta" + sbjct + ".tsv", fstream::out);
        removerFileDelta.open("./cppData/subject" + sbjct + "/remover_subject_Delta" + sbjct + ".tsv", fstream::out);
        weightFileDelta.open("./cppData/subject" + sbjct + "/lWeights_closed_subject_Delta" + sbjct + ".tsv", fstream::out);
        innerFileDelta.open("./cppData/subject" + sbjct + "/inner_subject_Delta" + sbjct + ".tsv", fstream::out);
        outerFileDelta.open("./cppData/subject" + sbjct + "/outer_subject_Delta" + sbjct + ".tsv", fstream::out);
        lmsFileDelta.open("./cppData/subject" + sbjct + "/lmsOutput_subject_Delta" + sbjct + ".tsv", fstream::out);
        lmsRemoverFileDelta.open("./cppData/subject" + sbjct + "/lmsCorrelation_subject_Delta" + sbjct + ".tsv", fstream::out);
        lmsFilterDelta = new Fir1(LMS_COEFF);
        lmsFilterDelta->setLearningRate(LMS_LEARNING_RATE);
        NND->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        double corrLMSDelta = 0;
        double lmsOutputDelta = 0;
        eegInfileDelta.open("./SubjectData/Delta/EEG_Subject" + sbjct + ".tsv");
        int lineNbDelta = 0;
        std::string lineDelta;
        while (!eegInfileDelta.eof())
        {
            std::getline(eegInfileDelta, lineDelta);
            lineNbDelta++;
        }
        cout << lineNbDelta << endl;
        progresscpp::ProgressBar progressBarDelta(lineNbDelta, 70);
        eegInfileDelta.close();
        eegInfileDelta.open("./SubjectData/Delta/EEG_Subject" + sbjct + ".tsv");
        cout << "Delta waves start:" << endl;
        while (!eegInfileDelta.eof())
        {
            ++progressBarDelta;
            progressBarDelta.display();
            count += 1;
            eegInfileDelta >>
                sampleNum >> innerDataDelta >> outerDataDelta;

            outerDataDelta *= outerGainDelta;
            innerDataDelta *= innerGainDelta;

            double innerFilteredDelta = innerFilter[0]->filter(innerDataDelta);
            innerDeltaDelayLine.push_back(innerFilteredDelta);
            double innerDelta = innerDeltaDelayLine[0];
            double outer = outerFilter[0]->filter(outerDataDelta);
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outerDelayLine[i] = outerDelayLine[i - 1];
            }
            outerDelayLine[0] = outer;
            double *outerDelayed = &outerDelayLine[0];
            NND->setInputs(outerDelayed);
            NND->propInputs();
            double removerDelta = NND->getOutput(0) * removerGainDelta;
            double fNNDelta = (innerDelta - removerDelta) * fnnGainDelta;
            // double snrFNN = pow(fNN, 2) / pow(outerData, 2);
            removerFileDelta
                << removerDelta << endl;
            nnFileDelta
                << fNNDelta << endl;
            NND->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NND->setBackwardError(fNNDelta);
            NND->propErrorBackward();
            // w /= count;
            NND->setLearningRate(wDelta, bDelta);
            NND->updateWeights();
            NND->snapWeights("cppData", "Delta", SUBJECT);
            corrLMSDelta += lmsFilterDelta->filter(outer);
            lmsOutputDelta = innerDelta - corrLMSDelta;
            lmsFilterDelta->lms_update(lmsOutputDelta);
            lmsFileDelta
                << lmsOutputDelta << endl;
            lmsRemoverFileDelta
                << corrLMSDelta << endl;
            innerFileDelta
                << innerDelta << endl;
            outerFileDelta << outer << endl;
        }
        progressBar.done();
        paramsFile << NLAYERS << " " << NLAYERS << endl;
        paramsFile << outerDelayLineLength << " " << outerDelayLineLength << endl;
        paramsFile << innerDelayLineLength << " " << innerDelayLineLength << endl;
        paramsFile << outerGainAlpha << " " << outerGainDelta << endl;
        paramsFile << innerGainAlpha << " " << innerGainDelta << endl;
        paramsFile << removerGainAlpha << " " << removerGainDelta << endl;
        paramsFile << fnnGainAlpha << " " << fnnGainDelta << endl;
        paramsFile << wAlpha << " " << wDelta << endl;
        paramsFile << bAlpha << " " << bDelta << endl;
        double endTime = time(nullptr);
        double duration = endTime - startTime;
        cout << "Duration: " << double(duration / 60.0) << " mins" << endl;
        closeFiles();
        cout << "The program has reached the end of the input file" << endl;
    }
}