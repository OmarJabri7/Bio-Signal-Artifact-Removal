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

fstream nnFile, removerFile, weightFile;
fstream innerFile;
fstream snrFile;
fstream outerFile;
fstream paramsFile;
fstream lmsFile;
fstream lmsRemoverFile;
fstream laplaceFile;
ifstream eegInfile;

int subjectsNb = 1;

int nNeurons[NLAYERS] = {N22, N21, N20, N19, N18, N17, N16, N15, N14, N13, N12, N11, N10, N9, N8, N7, N6, N5, N4, N3, N2, N1, N0};
int *numNeuronsP = nNeurons;
int numInputs = 1;
Net *NN = new Net(NLAYERS, numNeuronsP, numInputs, 0, "DNF");

std::vector<double> minErrors;

double sampleNum, outerData, innerData;
double newParam;
int paramCount, inParam = 0;
double errorNN;
double outerGain, innerGain, removerGain, fnnGain, w, b;
double outerDelayLine[outerDelayLineLength] = {0.0};
boost::circular_buffer<double> innerDelayLine(innerDelayLineLength);
int inputNum = outerDelayLineLength;
int main(int argc, const char *argv[])
{
    outerGain = 10.1;
    innerGain = 19.0;
    removerGain = 2.5;
    fnnGain = 0.5;
    w = 0.31622776601683794;
    b = 2.0;
    cout << "Remover gain: " << removerGain << endl;
    cout << "Feedback gain: " << fnnGain << endl;
    cout << "Weight ETA: " << w << endl;
    cout << "Bias ETA: " << b << endl;

    for (int s : tqdm::range(subjectsNb))
    {
        double startTime = time(NULL);
        int SUBJECT = s;
        int count = 0;
        cout << "subject: " << SUBJECT << endl;
        string sbjct = std::to_string(SUBJECT);
        nnFile.open("./cppData/subject" + sbjct + "/fnn_subject" + sbjct + ".tsv", fstream::out);
        removerFile.open("./cppData/subject" + sbjct + "/remover_subject" + sbjct + ".tsv", fstream::out);
        weightFile.open("./cppData/subject" + sbjct + "/lWeights_closed_subject" + sbjct + ".tsv", fstream::out);
        innerFile.open("./cppData/subject" + sbjct + "/inner_subject" + sbjct + ".tsv", fstream::out);
        outerFile.open("./cppData/subject" + sbjct + "/outer_subject" + sbjct + ".tsv", fstream::out);
        paramsFile.open("./cppData/subject" + sbjct + "/cppParams_subject" + sbjct + ".tsv", fstream::out);
        eegInfile.open("./SubjectData/EEG_Subject" + sbjct + ".tsv");
        // while (paramsFile.eof())
        // {
        //     paramsFile >> removerGain;
        //     paramsFile >> fnnGain;
        //     paramsFile >> w;
        // }
        NN->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
        double minError;

        while (!eegInfile.eof())
        {
            count += 1;
            eegInfile >>
                sampleNum >> innerData >> outerData;
            // if (paramCount < params.size())
            // {
            //     if (count >= 3000)
            //     {
            //         if (count % 6000 == 0.0)
            //         {
            //             cout << "Param NO: " << paramCount << endl;
            //             inParam = 0;
            //             cout << "Final Vector gains:" << endl;
            //             print_vector(errors);
            //             int idx = std::distance(errors.begin(), std::max_element(errors.begin(), errors.end()));
            //             errors = {};
            //             minErrors.push_back(minError);
            //             minError = 0;
            //             newParam = params[paramCount][idx];
            //             switch (paramCount)
            //             {
            //             case 0:
            //                 removerGain = newParam;
            //                 paramsFile << removerGain;
            //                 paramsFile << endl;
            //                 cout << paramCount << " new remover param: " << newParam << endl;
            //                 break;
            //             case 1:
            //                 fnnGain = newParam;
            //                 paramsFile << fnnGain;
            //                 paramsFile << endl;
            //                 cout << paramCount << " new fnn param: " << newParam << endl;
            //                 break;
            //             case 2:
            //                 w = newParam;
            //                 paramsFile << w;
            //                 paramsFile << endl;
            //                 cout << paramCount << " new weight param: " << newParam << endl;
            //                 break;
            //             case 3:
            //                 b = newParam;
            //                 paramsFile << b;
            //                 paramsFile << endl;
            //                 cout << paramCount << " new bias param: " << newParam << endl;
            //                 break;
            //             default:
            //                 break;
            //             }
            //             paramCount++;
            //         }
            //     }
            //     if (count > 500 && paramCount < params.size())
            //     {

            //         if (count % 300 == 0.0)
            //         {
            //             if (errorNN < minError)
            //             {
            //                 minError = errorNN;
            //             }
            //             errors.push_back(errorNN);
            //             inParam++;
            //             newParam = params[paramCount][inParam];
            //             switch (paramCount)
            //             {
            //             case 0:
            //                 removerGain = newParam;
            //                 break;
            //             case 1:
            //                 fnnGain = newParam;
            //                 break;
            //             case 2:
            //                 w = newParam;
            //                 break;
            //             case 3:
            //                 b = newParam;
            //                 break;
            //             default:
            //                 break;
            //             }
            //         }
            //     }
            // }
            outerData *= outerGain;
            innerData *= innerGain;
            innerDelayLine.push_back(innerData);
            double inner = innerDelayLine[0];
            for (int i = outerDelayLineLength - 1; i > 0; i--)
            {
                outerDelayLine[i] = outerDelayLine[i - 1];
            }
            outerDelayLine[0] = outerData;
            double *outerDelayed = &outerDelayLine[0];
            NN->setInputs(outerDelayed); // Here Input
            NN->propInputs();
            double removerNN = NN->getOutput(0) * removerGain;
            double fNN = (inner - removerNN) * fnnGain;
            errorNN = fNN;
            double snrFNN = pow(fNN, 2) / pow(outerData, 2);
            errorNN = snrFNN;
            removerFile
                << removerNN << endl;
            nnFile << fNN << endl;
            NN->setErrorCoeff(0, 1, 0, 0, 0, 0); //global, back, mid, forward, local, echo error
            NN->setBackwardError(fNN);
            NN->propErrorBackward();
            // w /= count;
            NN->setLearningRate(w, b);
            NN->updateWeights();
            NN->snapWeights("cppData", "DNF", SUBJECT);
        }
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
        eegInfile.close();
        cout << "The program has reached the end of the input file" << endl;
    }
}