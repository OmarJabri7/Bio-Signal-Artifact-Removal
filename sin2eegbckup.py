from eeg_generator import EEG_GEN
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "Data/experiment_data"
res_dir = "deepNeuronalFilter/SubjectData"
sbjcts = 4
noises = ["blink", "sudoku", "templerun", "blink+sudoku"]
EEG_GAIN = 1
NOISE_GAIN = 1
# ******************** Main ********************
# %%


def gen_eeg():
    noise_lengths = []
    eeg_obj = EEG_GEN()
    if(not os.path.exists("results.xlsx")):
        data_frame = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "Subject Number"])
        data_frame.to_excel(
            r"results.xlsx", sheet_name="DNF results", index=False)
    for sbj in range(sbjcts):
        for noise in noises:
            eeg_obj.load_data(dir, str(sbj), noise)
            noise_lengths.append(eeg_obj.get_noise_length(str(sbj), noise))
        eeg_obj.get_min_samples(noise_lengths)
        eeg_obj.gen_sin(A=1, f=10, fs=eeg_obj.fs)
        pure_eeg = eeg_obj.get_pure()
        eeg_obj.gen_eeg(str(sbj), noises[sbj])
        eeg = eeg_obj.get_eeg(str(sbj), noises[sbj])
        # eeg = eeg_obj.butter_bandpass(eeg, 1, 100, fs=eeg_obj.fs, order=2)
        noise = eeg_obj.get_noise(str(sbj), noises[sbj])
        NOISE_GAIN = 1
        EEG_GAIN = 1
        noise *= NOISE_GAIN
        eeg *= EEG_GAIN
        print(NOISE_GAIN)
        print(EEG_GAIN)
        # eeg_obj.save_data(eeg, noise, str(sbj))
        eeg_obj.plot_time_series(pure_eeg[1000:2000],
                                 f"Pure Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(pure_eeg, eeg_obj.fs,
                               f"Pure Frequency Subject {sbj}")
        plt.figure()
        eeg_obj.plot_time_series(eeg[1000:2000],
                                 f"Original Temporal Subject {sbj}")
        eeg_obj.plot_time_series(noise,
                                 f"Noise Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg, eeg_obj.fs,
                               f"Original Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(noise, eeg_obj.fs,
                               f"Noise Frequency Subject {sbj}")
        plt.show()
        print(10*np.log10(eeg_obj.calc_SNR(eeg, noise, 0)))


def get_results():
    eeg_obj = EEG_GEN()
    results_df = pd.read_excel(r"results.xlsx", "DNF results")
    for sbj in range(sbjcts):
        dnf_res = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject{sbj}.tsv")
        lms_res = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject{sbj}.tsv")
        remover_res = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject{sbj}.tsv")
        data = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/EEG_Subject{sbj}.tsv")
        inner = data[:, 1]
        outer = data[:, 2]
        # dnf_res = eeg_obj.butter_bandpass(dnf_res, 1, 300, eeg_obj.fs, 2)
        parameters = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/cppParams_subject{sbj}.tsv")
        SNR_ORIG = 10*np.log10(eeg_obj.calc_SNR(inner, outer, 1))
        if(SNR_ORIG == None):
            SNR_ORIG = -np.inf
        SNR_DNF = 10*np.log10(eeg_obj.calc_SNR(dnf_res, outer, 1))
        SNR_LMS = (eeg_obj.calc_SNR(lms_res, outer, 1))
        print(SNR_ORIG)
        print(SNR_DNF)
        print(SNR_LMS)
        eeg_obj.plot_time_series(inner[1000:2000],
                                 f"Original Temporal Subject {sbj}")
        eeg_obj.plot_time_series(dnf_res[1000:2000],
                                 f"DNF Temporal Subject {sbj}")
        eeg_obj.plot_time_series(lms_res[1000:2000],
                                 f"LMS Temporal Subject {sbj}")
        eeg_obj.plot_time_series(outer,
                                 f"Noise Temporal Subject {sbj}")
        eeg_obj.plot_time_series(remover_res,
                                 f"Remover Temporal Subject {sbj}")
        eeg_obj.plot_welch(inner, eeg_obj.fs,
                           title=f"Noise Frequency Subject {sbj}")
        eeg_obj.plot_welch(dnf_res, eeg_obj.fs,
                           title=f"Noise Frequency Subject {sbj}")
        plt.figure()
        eeg_obj.plot_freq_resp(outer, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(lms_res, Fs=eeg_obj.fs,
                               title=f"LMS Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(remover_res, Fs=eeg_obj.fs,
                               title=f"Remover Frequency Subject {sbj}")
        plt.show()
        parameters = np.append(parameters, EEG_GAIN)
        parameters = np.append(parameters, NOISE_GAIN)
        parameters = np.append(parameters, SNR_ORIG)
        parameters = np.append(parameters, SNR_DNF)
        parameters = np.append(parameters, sbj)
        results_df = results_df.append(pd.DataFrame(
            [parameters], columns=results_df.columns), ignore_index=True)
        results_df.to_excel("results.xlsx", "DNF results", index=False)


gen_eeg()
# get_results()
