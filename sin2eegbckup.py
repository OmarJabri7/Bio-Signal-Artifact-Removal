from eeg_generator import EEG_GEN
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook

dir = "Data/experiment_data"
res_dir = "deepNeuronalFilter/SubjectData"
sbjcts = 4
noises = ["blink", "sudoku", "templerun", "blink+sudoku"]
ALPHA_GAIN = 1
DELTA_GAIN = 1
NOISE_GAIN = 1
DELTA_LOW = 1
DELTA_HIGH = 6
ALPHA_LOW = 8
ALPHA_HIGH = 14
# ******************** Main ********************
# %%


def gen_eeg():
    noise_lengths = []
    eeg_obj = EEG_GEN()
    alpha = range(ALPHA_LOW, ALPHA_HIGH)
    delta = range(DELTA_LOW, DELTA_HIGH)
    if(not os.path.exists("results.xlsx")):
        data_frame_alpha = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "Subject Number"])
        data_frame_delta = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "Subject Number"])
        eeg_obj.save_xls([data_frame_alpha, data_frame_delta],
                         "results.xlsx", ["Alpha", "Delta"])
    for sbj in range(sbjcts):
        for noise in noises:
            eeg_obj.load_data(dir, str(sbj), noise)
            noise_lengths.append(eeg_obj.get_noise_length(str(sbj), noise))
        eeg_obj.get_min_samples(noise_lengths)
        eeg_obj.gen_sine_alpha(A=1, f=10, freqs=alpha, fs=eeg_obj.fs)
        pure_eeg_alpha = eeg_obj.get_pure_alpha()
        eeg_obj.gen_sine_delta(A=0.3, f=3, freqs=delta, fs=eeg_obj.fs)
        pure_eeg_delta = eeg_obj.get_pure_delta()
        eeg_obj.gen_eeg_alpha(str(sbj), noises[sbj])
        eeg_obj.gen_eeg_delta(str(sbj), noises[sbj])
        eeg_alpha = eeg_obj.get_eeg_alpha(str(sbj), noises[sbj])
        eeg_delta = eeg_obj.get_eeg_delta(str(sbj), noises[sbj])
        noise = eeg_obj.get_noise(str(sbj), noises[sbj])
        NOISE_GAIN = eeg_obj.gen_gain(noise)
        ALPHA_GAIN = eeg_obj.gen_gain(eeg_alpha)
        DELTA_GAIN = eeg_obj.gen_gain(eeg_delta)
        print(NOISE_GAIN)
        print(DELTA_GAIN)
        print(ALPHA_GAIN)
        noise *= NOISE_GAIN
        eeg_alpha *= ALPHA_GAIN
        eeg_delta *= DELTA_GAIN
        eeg_obj.save_data(eeg_alpha, noise, str(sbj), "Alpha")
        eeg_obj.save_data(eeg_delta, noise, str(sbj), "Delta")
        eeg_obj.plot_time_series(pure_eeg_alpha[1000:2000],
                                 f"Pure Alpha Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(pure_eeg_alpha, eeg_obj.fs,
                               f"Pure Alpha Frequency Subject {sbj}")
        eeg_obj.plot_time_series(pure_eeg_delta[1000:2000],
                                 f"Pure Delta Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(pure_eeg_delta, eeg_obj.fs,
                               f"Pure Delta Frequency Subject {sbj}")
        eeg_obj.plot_time_series(eeg_alpha[1000:2000],
                                 f"Noisy Alpha Temporal Subject {sbj}")
        eeg_obj.plot_time_series(eeg_delta[1000:2000],
                                 f"Noisy Delta Temporal Subject {sbj}")
        eeg_obj.plot_time_series(noise,
                                 f"Noise Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"Noisy Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"Noisy Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(noise, eeg_obj.fs,
                               f"Noise Frequency Subject {sbj}")
        plt.show()
        print(
            f"Pure Alpha: {10*np.log10(eeg_obj.calc_SNR(pure_eeg_alpha, noise, 0, 0))} dB")
        print(
            f"Pure Delta: {10*np.log10(eeg_obj.calc_SNR(pure_eeg_delta, noise, 0, 0 ))} dB")
        print(
            f"Noisy Alpha: {10*np.log10(eeg_obj.calc_SNR(eeg_alpha, noise, 1, 0))} dB")
        print(
            f"Noisy Delta: {10*np.log10(eeg_obj.calc_SNR(eeg_delta, noise, 1, 0))} dB")


def get_results():
    eeg_obj = EEG_GEN()
    results_df_alpha = pd.read_excel(r"results.xlsx", "Alpha")
    results_df_delta = pd.read_excel(r"results.xlsx", "Delta")
    for sbj in range(sbjcts):
        dnf = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject{sbj}.tsv")
        dnf_alpha = dnf[:, 0]
        dnf_delta = dnf[:, 1]
        lms = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject{sbj}.tsv")
        lms_alpha = lms[:, 0]
        lms_delta = lms[:, 1]
        remover = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject{sbj}.tsv")
        remover_alpha = remover[:, 0]
        remover_delta = remover[:, 1]
        alpha_data = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/Alpha/EEG_Subject{sbj}.tsv")
        delta_data = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/Delta/EEG_Subject{sbj}.tsv")
        inner_alpha = alpha_data[:, 1]
        outer_alpha = alpha_data[:, 2]
        inner_delta = delta_data[:, 1]
        outer_delta = delta_data[:, 2]
        NOISE_GAIN = eeg_obj.gen_gain(outer_alpha)
        ALPHA_GAIN = eeg_obj.gen_gain(inner_alpha)
        DELTA_GAIN = eeg_obj.gen_gain(inner_delta)
        parameters = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/cppParams_subject{sbj}.tsv")
        alpha_params = parameters[:, 0]
        delta_params = parameters[:, 1]
        SNR_ORIG_ALPHA = 10 * \
            np.log10(eeg_obj.calc_SNR(inner_alpha, outer_alpha, 1, 0))
        SNR_ORIG_DELTA = 10 * \
            np.log10(eeg_obj.calc_SNR(inner_delta, outer_delta, 1, 0))
        if(SNR_ORIG_ALPHA == None):
            SNR_ORIG_ALPHA = -np.inf
        SNR_DNF_ALPHA = 10 * \
            np.log10(eeg_obj.calc_SNR(dnf_alpha, outer_alpha, 1, 0))
        SNR_LMS_ALPHA = (eeg_obj.calc_SNR(lms_alpha, outer_alpha, 1, 0))
        SNR_DNF_DELTA = 10 * \
            np.log10(eeg_obj.calc_SNR(dnf_delta, outer_delta, 1, 0))
        SNR_LMS_DELTA = (eeg_obj.calc_SNR(lms_delta, outer_delta, 1, 0))
        print("Alpha Results: ")
        print(SNR_ORIG_ALPHA)
        print(SNR_DNF_ALPHA)
        print(SNR_LMS_ALPHA)
        print("Delta Results: ")
        print(SNR_ORIG_DELTA)
        print(SNR_DNF_DELTA)
        print(SNR_LMS_DELTA)
        eeg_obj.plot_time_series(inner_alpha,
                                 f"Original Temporal Subject {sbj}")
        eeg_obj.plot_time_series(dnf_alpha,
                                 f"DNF Alpha Temporal Subject {sbj}")
        eeg_obj.plot_time_series(lms_alpha,
                                 f"LMS Alpha Temporal Subject {sbj}")
        eeg_obj.plot_time_series(outer_alpha,
                                 f"Noise Temporal Subject {sbj}")
        eeg_obj.plot_time_series(remover_alpha,
                                 f"Remover Alpha Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(inner_alpha, eeg_obj.fs,
                               title=f"Noisy Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(dnf_alpha, eeg_obj.fs,
                               title=f"DNF Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(outer_alpha, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(lms_alpha, Fs=eeg_obj.fs,
                               title=f"LMS Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(remover_alpha, Fs=eeg_obj.fs,
                               title=f"Remover Alpha Frequency Subject {sbj}")

        eeg_obj.plot_time_series(inner_delta,
                                 f"Original Delta Temporal Subject {sbj}")
        eeg_obj.plot_time_series(dnf_delta,
                                 f"DNF Delta Temporal Subject {sbj}")
        eeg_obj.plot_time_series(lms_delta,
                                 f"LMS Delta Temporal Subject {sbj}")
        eeg_obj.plot_time_series(outer_delta,
                                 f"Noise Temporal Subject {sbj}")
        eeg_obj.plot_time_series(remover_delta,
                                 f"Remover Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(inner_delta, eeg_obj.fs,
                               title=f"Noisy Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(dnf_delta, eeg_obj.fs,
                               title=f"DNF Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(outer_delta, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(lms_delta, Fs=eeg_obj.fs,
                               title=f"LMS Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(remover_delta, Fs=eeg_obj.fs,
                               title=f"Remover Delta Frequency Subject {sbj}")
        plt.show()
        alpha_params = np.append(alpha_params, ALPHA_GAIN)
        alpha_params = np.append(alpha_params, NOISE_GAIN)
        alpha_params = np.append(alpha_params, SNR_ORIG_ALPHA)
        alpha_params = np.append(alpha_params, SNR_DNF_ALPHA)
        alpha_params = np.append(alpha_params, sbj)
        delta_params = np.append(delta_params, DELTA_GAIN)
        delta_params = np.append(delta_params, NOISE_GAIN)
        delta_params = np.append(delta_params, SNR_ORIG_DELTA)
        delta_params = np.append(delta_params, SNR_DNF_DELTA)
        delta_params = np.append(delta_params, sbj)
        results_df_alpha = results_df_alpha.append(pd.DataFrame(
            [alpha_params], columns=results_df_alpha.columns), ignore_index=True)
        results_df_delta = results_df_delta.append(pd.DataFrame(
            [delta_params], columns=results_df_delta.columns), ignore_index=True)
        writer = pd.ExcelWriter("results.xlsx", engine='xlsxwriter')
        results_df_alpha.to_excel(writer, "Alpha", index=False)
        results_df_delta.to_excel(writer, "Delta", index=False)
        writer.save()
        writer.close()


gen_eeg()
# get_results()
