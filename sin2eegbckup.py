from eeg_generator import EEG_GEN
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook

dir = "Data/experiment_data"
res_dir = "deepNeuronalFilter/SubjectData"
sbjcts = 3
noises_alpha = ["eyescrunching", "jaw",
                "turninghead"]  # "jaw+eyescrunching"]
noises_delta = ["blink", "sudoku", "templerun"]  # , "sudoku+blink"]
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
    eeg_obj = EEG_GEN()
    alpha = range(ALPHA_LOW, ALPHA_HIGH)
    delta = range(DELTA_LOW, DELTA_HIGH)
    if(not os.path.exists("results.xlsx")):
        data_frame_alpha = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta", "Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "SNR LMS", "Subject Number", "Noise"])
        data_frame_delta = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta", "Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "SNR LMS", "Subject Number", "Noise"])
        eeg_obj.save_xls([data_frame_alpha, data_frame_delta],
                         "results.xlsx", ["Alpha", "Delta"])
    params_df_alpha = pd.DataFrame(
        columns=["Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Noise Gain", "Signal Gain", "Pure SNR",  "Subject Number", "Noise"])
    params_df_delta = pd.DataFrame(
        columns=["Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Noise Gain", "Signal Gain", "Pure SNR",  "Subject Number", "Noise"])
    eeg_obj.save_xls([params_df_alpha, params_df_delta],
                     "initial_params.xlsx", ["Alpha", "Delta"])
    for sbj in range(sbjcts):
        params_alpha = []
        params_delta = []
        eeg_obj.load_alpha_data(dir, str(sbj), noises_alpha[sbj])
        eeg_obj.load_delta_data(dir, str(sbj), noises_delta[sbj])
        noise_alpha = eeg_obj.get_noise(str(sbj), noises_alpha[sbj])
        noise_delta = eeg_obj.get_noise(str(sbj), noises_delta[sbj])
        A_alpha = 0.1
        f_alpha = 10
        sum_sines_alpha = False
        optimal_alpha = False
        bandpass_alpha = False
        eeg_obj.gen_sine_alpha(A=A_alpha, f=f_alpha, freqs=alpha,
                               fs=eeg_obj.fs, bandpass=bandpass_alpha, sum_sines=sum_sines_alpha, optimal=optimal_alpha)
        pure_eeg_alpha = eeg_obj.get_pure_alpha()
        A_delta = 0.1
        f_delta = 3
        sum_sines_delta = False
        optimal_delta = False
        bandpass_delta = False
        eeg_obj.gen_sine_delta(A=A_delta, f=f_delta, freqs=delta,
                               fs=eeg_obj.fs, bandpass=bandpass_delta, sum_sines=sum_sines_delta, optimal=optimal_delta)
        pure_eeg_delta = eeg_obj.get_pure_delta()
        eeg_obj.gen_eeg_alpha(str(sbj), noises_alpha[sbj])
        eeg_obj.gen_eeg_delta(str(sbj), noises_delta[sbj])
        eeg_alpha = eeg_obj.get_eeg_alpha(str(sbj), noises_alpha[sbj])
        eeg_alpha = eeg_obj.butter_highpass(eeg_alpha, 1, eeg_obj.fs, 4)
        eeg_alpha = eeg_obj.butter_bandstop(
            eeg_alpha, 49, 51, eeg_obj.fs, 2)
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"Before LowPass Noisy Alpha Frequency Subject {sbj}")
        eeg_alpha = eeg_obj.butter_lowpass(
            eeg_alpha, 100, eeg_obj.fs, 2)
        eeg_delta = eeg_obj.get_eeg_delta(str(sbj), noises_delta[sbj])
        eeg_delta = eeg_obj.butter_highpass(eeg_delta, 1, eeg_obj.fs, 4)
        eeg_delta = eeg_obj.butter_bandstop(
            eeg_delta, 49, 51, eeg_obj.fs, 2)
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"Before LowPass Noisy Delta Frequency Subject {sbj}")
        eeg_delta = eeg_obj.butter_lowpass(
            eeg_delta, 100, eeg_obj.fs, 2)
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"After LowPass Noisy Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"After LowPass Noisy Delta Frequency Subject {sbj}")
        ALPHA_GAIN = eeg_obj.gen_gain(eeg_alpha)
        NOISE_GAIN_ALPHA = eeg_obj.gen_gain(noise_alpha)
        NOISE_GAIN_DELTA = eeg_obj.gen_gain(noise_delta)
        DELTA_GAIN = eeg_obj.gen_gain(eeg_delta)
        pure_alpha_SNR = 10 * \
            np.log10(eeg_obj.calc_SNR(pure_eeg_alpha, noise_alpha, 0, 1, 0))
        pure_delta_SNR = 10 * \
            np.log10(eeg_obj.calc_SNR(pure_eeg_delta, noise_delta, 1, 1, 0))
        # noise_alpha *= NOISE_GAIN_ALPHA
        # noise_delta *= NOISE_GAIN_DELTA
        # eeg_alpha *= ALPHA_GAIN
        # eeg_delta *= DELTA_GAIN
        print(
            f"Pure Alpha: {pure_alpha_SNR} dB")
        print(
            f"Pure Delta: {pure_delta_SNR} dB")
        eeg_obj.save_data(eeg_alpha, noise_alpha, str(sbj), "Alpha")
        eeg_obj.save_data(eeg_delta, noise_delta, str(sbj), "Delta")
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
        eeg_obj.plot_time_series(noise_alpha,
                                 f"Noise Alpha Temporal Subject {sbj}")
        eeg_obj.plot_time_series(noise_delta,
                                 f"Noise Delta Temporal Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"Noisy Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"Noisy Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(noise_alpha, eeg_obj.fs,
                               f"Noise Alpha Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp(noise_delta, eeg_obj.fs,
                               f"Noise Delta Frequency Subject {sbj}")
        eeg_obj.plot_freq_resp_vs(0, pure_eeg_alpha, eeg_alpha, eeg_obj.fs,
                                  f"Pure vs Noisy Alphas Frequency Subject {sbj}, Noise: {noises_alpha[sbj]}", "Pure Alpha", "Noisy Alpha")
        eeg_obj.plot_freq_resp_vs(1, pure_eeg_delta, eeg_delta, eeg_obj.fs,
                                  f"Pure vs Noisy Deltas Frequency Subject {sbj}, Noise: {noises_delta[sbj]}", "Pure Delta", "Noisy Delta")
        # plt.show()
        params_alpha.append([A_alpha, f_alpha, sum_sines_alpha, bandpass_alpha,
                            optimal_alpha, NOISE_GAIN_ALPHA, ALPHA_GAIN, pure_alpha_SNR, sbj, noises_alpha[sbj]])
        params_delta.append([A_delta, f_delta, sum_sines_delta, bandpass_delta,
                            optimal_delta, NOISE_GAIN_DELTA, DELTA_GAIN, pure_delta_SNR, sbj, noises_delta[sbj]])
        params_df_alpha = params_df_alpha.append(pd.DataFrame(
            [params_alpha[0]], columns=params_df_alpha.columns), ignore_index=True)
        params_df_delta = params_df_delta.append(pd.DataFrame(
            [params_delta[0]], columns=params_df_delta.columns), ignore_index=True)
        writer = pd.ExcelWriter("initial_params.xlsx", engine='xlsxwriter')
        params_df_alpha.to_excel(writer, "Alpha", index=False)
        params_df_delta.to_excel(writer, "Delta", index=False)
        writer.save()
        writer.close()


def get_results():
    eeg_obj = EEG_GEN()
    results_df_alpha = pd.read_excel(r"results.xlsx", "Alpha")
    results_df_delta = pd.read_excel(r"results.xlsx", "Delta")
    init_df_alpha = pd.read_excel(r"initial_params.xlsx", "Alpha")
    init_df_delta = pd.read_excel(r"initial_params.xlsx", "Delta")
    for sbj in range(sbjcts):
        params_sbj_alpha = init_df_alpha.loc[init_df_alpha['Subject Number'] == sbj]
        params_sbj_delta = init_df_delta.loc[init_df_delta['Subject Number'] == sbj]
        dnf_alpha = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject_Alpha{sbj}.tsv")
        dnf_delta = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject_Delta{sbj}.tsv")
        NOISE_GAIN_ALPHA = params_sbj_alpha["Noise Gain"].values
        NOISE_GAIN_DELTA = params_sbj_delta["Noise Gain"].values
        ALPHA_GAIN = params_sbj_alpha["Signal Gain"].values
        DELTA_GAIN = params_sbj_delta["Signal Gain"].values
        # dnf_alpha = dnf[:, 0]
        # dnf_alpha /= ALPHA_GAIN[0]
        # dnf_delta = dnf[:, 1]
        # dnf_delta /= DELTA_GAIN[0]
        # dnf_alpha = eeg_obj.butter_highpass(dnf_alpha, 1, eeg_obj.fs, 1)
        # dnf_alpha = eeg_obj.butter_bandstop(dnf_alpha, 1, 5, eeg_obj.fs, 5)
        dnf_delta = eeg_obj.butter_highpass(dnf_delta, 1, eeg_obj.fs, 2)
        # dnf_delta = eeg_obj.butter_bandstop(dnf_delta, 8, 12, eeg_obj.fs, 4)
        lms_alpha = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject_Alpha{sbj}.tsv")
        lms_delta = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject_Delta{sbj}.tsv")
        # lms_alpha = lms[:, 0]
        # lms_delta = lms[:, 1]
        lms_alpha = eeg_obj.butter_highpass(lms_alpha, 1, eeg_obj.fs, 2)
        lms_delta = eeg_obj.butter_highpass(lms_delta, 1, eeg_obj.fs, 2)
        remover_alpha = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject_Alpha{sbj}.tsv")
        remover_delta = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject_Delta{sbj}.tsv")
        # remover_alpha = remover[:, 0]
        # remover_delta = remover[:, 1]
        alpha_data = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/Alpha/EEG_Subject{sbj}.tsv")
        delta_data = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/Delta/EEG_Subject{sbj}.tsv")
        inner_alpha = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/inner_subject_Alpha{sbj}.tsv")
        # inner_alpha = alpha_data[:, 1]
        outer_alpha = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/outer_subject_Alpha{sbj}.tsv")
        # outer_alpha = alpha_data[:, 2]
        # outer_alpha /= NOISE_GAIN_ALPHA[0]
        inner_delta = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/inner_subject_Delta{sbj}.tsv")
        # inner_delta = delta_data[:, 1]
        outer_delta = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/outer_subject_Delta{sbj}.tsv")
        # outer_delta = delta_data[:, 2]
        # outer_delta /= NOISE_GAIN_DELTA[0]
        # parameters = np.loadtxt(
        #     f"deepNeuronalFilter/cppData/subject{sbj}/cppParams_subject{sbj}.tsv")
        # alpha_params = parameters[:, 0]
        # delta_params = parameters[:, 1]
        snr_fct = 1
        calc = 1
        alpha = 0
        delta = 1
        SNR_ORIG_ALPHA = params_sbj_alpha["Pure SNR"].values
        SNR_ORIG_ALPHA = SNR_ORIG_ALPHA[0]
        SNR_ORIG_DELTA = params_sbj_delta["Pure SNR"].values
        SNR_ORIG_DELTA = SNR_ORIG_DELTA[0]
        SNR_DNF_ALPHA = 10 * \
            np.log10(eeg_obj.calc_SNR(
                dnf_alpha, outer_alpha, alpha, snr_fct, calc))
        SNR_LMS_ALPHA = (eeg_obj.calc_SNR(
            lms_alpha, outer_alpha, alpha, snr_fct, calc))
        SNR_DNF_DELTA = 10 * \
            np.log10(eeg_obj.calc_SNR(
                dnf_delta, outer_delta, delta, snr_fct, calc))
        SNR_LMS_DELTA = (eeg_obj.calc_SNR(
            lms_delta, outer_delta, delta, snr_fct, calc))
        print("Alpha Results: ")
        print(SNR_ORIG_ALPHA)
        print(SNR_DNF_ALPHA)
        print(SNR_LMS_ALPHA)
        print("Delta Results: ")
        print(SNR_ORIG_DELTA)
        print(SNR_DNF_DELTA)
        print(SNR_LMS_DELTA)
        eeg_obj.plot_time_series(inner_alpha[1000:2000],
                                 f"Noisy Alpha Temporal Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_time_series(dnf_alpha[1000:2000],
                                 f"DNF Alpha Temporal Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_time_series(lms_alpha[1000:2000],
                                 f"LMS Alpha Temporal Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_time_series(outer_alpha,
                                 f"Noise Temporal Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_time_series(remover_alpha,
                                 f"Remover Alpha Temporal Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")

        eeg_obj.plot_freq_resp(inner_alpha, Fs=eeg_obj.fs,
                               title=f"Noisy Alpha Frequency Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_freq_resp_vs(
            alpha, inner_alpha, dnf_alpha, eeg_obj.fs, "Noisy vs Denoised alphas", "Noisy Alpha", "Denoised Alpha")
        eeg_obj.plot_freq_resp(outer_alpha, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_freq_resp(lms_alpha, Fs=eeg_obj.fs,
                               title=f"LMS Alpha Frequency Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")
        eeg_obj.plot_freq_resp(remover_alpha, Fs=eeg_obj.fs,
                               title=f"Remover Alpha Frequency Subject {sbj}, Noise: {params_sbj_alpha['Noise'].values}")

        eeg_obj.plot_time_series(inner_delta[1000:2000],
                                 f"Noisy Delta Temporal Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_time_series(dnf_delta[1000:2000],
                                 f"DNF Delta Temporal Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_time_series(lms_delta[1000:2000],
                                 f"LMS Delta Temporal Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_time_series(outer_delta,
                                 f"Noise Temporal Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_time_series(remover_delta,
                                 f"Remover Temporal Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_freq_resp(inner_delta, Fs=eeg_obj.fs,
                               title=f"Noisy Delta Frequency Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_freq_resp_vs(
            delta, inner_delta, dnf_delta, eeg_obj.fs, "Noisy vs Denoised deltas", "Noisy Delta", "Denoised Delta")
        eeg_obj.plot_freq_resp(outer_delta, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_freq_resp(lms_delta, Fs=eeg_obj.fs,
                               title=f"LMS Delta Frequency Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        eeg_obj.plot_freq_resp(remover_delta, Fs=eeg_obj.fs,
                               title=f"Remover Delta Frequency Subject {sbj}, Noise: {params_sbj_delta['Noise'].values}")
        plt.show()
        # alpha_params = np.append(
        #     alpha_params, params_sbj_alpha["Amplitude Sig"])
        # alpha_params = np.append(
        #     alpha_params, params_sbj_alpha["Frequency Sig"])
        # alpha_params = np.append(
        #     alpha_params, params_sbj_alpha["Sum Sinusoids"])
        # alpha_params = np.append(
        #     alpha_params, params_sbj_alpha["Bandpass Sinusoids"])
        # alpha_params = np.append(
        #     alpha_params, params_sbj_alpha["Optimal Sinusoids"])
        # alpha_params = np.append(alpha_params, ALPHA_GAIN)
        # alpha_params = np.append(alpha_params, NOISE_GAIN)
        # alpha_params = np.append(alpha_params, params_sbj_alpha["Pure SNR"])
        # alpha_params = np.append(alpha_params, SNR_ORIG_ALPHA)
        # alpha_params = np.append(alpha_params, SNR_DNF_ALPHA)
        # alpha_params = np.append(alpha_params, SNR_LMS_ALPHA)
        # alpha_params = np.append(alpha_params, sbj)
        # alpha_params = np.append(alpha_params, params_sbj_alpha["Noise"])

        # delta_params = np.append(
        #     delta_params, params_sbj_delta["Amplitude Sig"])
        # delta_params = np.append(
        #     delta_params, params_sbj_delta["Frequency Sig"])
        # delta_params = np.append(
        #     delta_params, params_sbj_delta["Sum Sinusoids"])
        # delta_params = np.append(
        #     delta_params, params_sbj_delta["Bandpass Sinusoids"])
        # delta_params = np.append(
        #     delta_params, params_sbj_delta["Optimal Sinusoids"])
        # delta_params = np.append(delta_params, DELTA_GAIN)
        # delta_params = np.append(delta_params, NOISE_GAIN)
        # delta_params = np.append(delta_params, params_sbj_delta["Pure SNR"])
        # delta_params = np.append(delta_params, SNR_ORIG_DELTA)
        # delta_params = np.append(delta_params, SNR_DNF_DELTA)
        # delta_params = np.append(delta_params, SNR_LMS_DELTA)
        # delta_params = np.append(delta_params, sbj)
        # delta_params = np.append(delta_params, params_sbj_delta["Noise"])
        # results_df_alpha = results_df_alpha.append(pd.DataFrame(
        #     [alpha_params], columns=results_df_alpha.columns), ignore_index=True)
        # results_df_delta = results_df_delta.append(pd.DataFrame(
        #     [delta_params], columns=results_df_delta.columns), ignore_index=True)
        # writer = pd.ExcelWriter("results.xlsx", engine='xlsxwriter')
        # results_df_alpha.to_excel(writer, "Alpha", index=False)
        # results_df_delta.to_excel(writer, "Delta", index=False)
        # writer.save()
        # writer.close()


gen_eeg()
# get_results()
