from eeg_generator import EEG_GEN
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "Data/experiment_data"
res_dir = "deepNeuronalFilter/SubjectData"
noises = 5
data_sbj = 4
noises_alpha_combos = ["eyescrunching+jaw", "jaw+raisingeyebrows"]
noises_delta_combos = ["blink+templerun", "blink+sudoku"]
noises_alpha = ["eyescrunching", "jaw",
                "raisingeyebrows", "movehat", "movehead"]
noises_delta = ["blink", "templerun", "sudoku", "flow", "wordsearch"]
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
    if(not os.path.exists("Results-Generation/subj{data_sbj}/results_Alpha.xlsx") and not os.path.exists("Results-Generation/subj{data_sbj}/results_Delta.xlsx")):
        data_frame_alpha = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta", "Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "SNR LMS", "Subject Number", "Noise"])
        data_frame_delta = pd.DataFrame(
            columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta", "Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF", "SNR LMS", "Subject Number", "Noise"])
        eeg_obj.save_xls([data_frame_alpha],
                         f"Results-Generation/subj{data_sbj}/results_Alpha.xlsx", ["DNF Results"])
        eeg_obj.save_xls([data_frame_delta],
                         f"Results-Generation/subj{data_sbj}/results_Delta.xlsx", ["DNF Results"])
    params_df_alpha = pd.DataFrame(
        columns=["Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Noise Gain", "Signal Gain", "Pure SNR",  "Subject Number", "Noise"])
    params_df_delta = pd.DataFrame(
        columns=["Amplitude Sig", "Frequency Sig", "Sum Sinusoids", "Bandpass Sinusoids", "Optimal Sinusoids", "Noise Gain", "Signal Gain", "Pure SNR",  "Subject Number", "Noise"])
    eeg_obj.save_xls([params_df_alpha, params_df_delta],
                     "initial_params.xlsx", ["Alpha", "Delta"])
    for sbj in range(noises):
        params_alpha = []
        params_delta = []
        eeg_obj.load_alpha_data(
            dir, str(data_sbj), str(sbj), noises_alpha[sbj])
        eeg_obj.load_delta_data(
            dir, str(data_sbj), str(sbj), noises_delta[sbj])
        noise_alpha = eeg_obj.get_noise(str(sbj), noises_alpha[sbj])
        noise_alpha = noise_alpha*1e2
        noise_delta = eeg_obj.get_noise(str(sbj), noises_delta[sbj])
        noise_delta = noise_delta*1e2
        eeg_obj.set_noise(str(sbj), noises_alpha[sbj], noise_alpha)
        eeg_obj.set_noise(str(sbj), noises_delta[sbj], noise_delta)
        A_alpha = 15  # micro Volts
        f_alpha = 10  # Hz
        sum_sines_alpha = False
        optimal_alpha = False
        bandpass_alpha = False
        eeg_obj.gen_sine_alpha(A=A_alpha, f=f_alpha, freqs=alpha,
                               fs=eeg_obj.fs, bandpass=bandpass_alpha, sum_sines=sum_sines_alpha, optimal=optimal_alpha)
        pure_eeg_alpha = eeg_obj.get_pure_alpha()
        A_delta = 15  # micro Volts
        f_delta = 3  # Hz
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
                               f"Before LowPass Noisy Alpha Frequency Subject {data_sbj}", data_sbj)
        eeg_alpha = eeg_obj.butter_lowpass(
            eeg_alpha, 100, eeg_obj.fs, 2)
        eeg_delta = eeg_obj.get_eeg_delta(str(sbj), noises_delta[sbj])
        eeg_delta = eeg_obj.butter_highpass(eeg_delta, 1, eeg_obj.fs, 4)
        eeg_delta = eeg_obj.butter_bandstop(
            eeg_delta, 49, 51, eeg_obj.fs, 2)
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"Before LowPass Noisy Delta Frequency Subject {data_sbj}", data_sbj)
        eeg_delta = eeg_obj.butter_lowpass(
            eeg_delta, 100, eeg_obj.fs, 2)
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"After LowPass Noisy Alpha Frequency Subject {data_sbj}", data_sbj)
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"After LowPass Noisy Delta Frequency Subject {data_sbj}", data_sbj)
        ALPHA_GAIN = eeg_obj.gen_gain(np.max(eeg_alpha))
        NOISE_GAIN_ALPHA = eeg_obj.gen_gain(np.max(noise_alpha))
        NOISE_GAIN_DELTA = eeg_obj.gen_gain(np.max(noise_delta))
        DELTA_GAIN = eeg_obj.gen_gain(np.max(eeg_delta))
        snr_fct = 1
        calc = 1
        pure_alpha_SNR = 10*np.log10(eeg_obj.calc_SNR(
            pure_eeg_alpha, eeg_alpha, 0, snr_fct, calc))
        pure_delta_SNR = 10*np.log10(eeg_obj.calc_SNR(
            pure_eeg_delta, eeg_delta, 1, snr_fct, calc))
        eeg_obj.plot_time_series(pure_eeg_alpha[1000:2000],
                                 f"Pure Alpha Temporal Subject {data_sbj}", data_sbj)
        eeg_obj.plot_freq_resp(pure_eeg_alpha, eeg_obj.fs,
                               f"Pure Alpha Frequency Subject {data_sbj}", data_sbj)
        eeg_obj.plot_time_series(pure_eeg_delta[1000:2000],
                                 f"Pure Delta Temporal Subject {data_sbj}", data_sbj)
        eeg_obj.plot_freq_resp(pure_eeg_delta, eeg_obj.fs,
                               f"Pure Delta Frequency Subject {data_sbj}", data_sbj)
        eeg_obj.plot_time_series(eeg_alpha[1000:2000],
                                 f"Noisy Alpha Temporal Subject {data_sbj}, Noise: {noises_alpha[sbj]}", data_sbj)
        eeg_obj.plot_time_series(eeg_delta[1000:2000],
                                 f"Noisy Delta Temporal Subject {data_sbj}, Noise: {noises_delta[sbj]}", data_sbj)
        eeg_obj.plot_time_series(noise_alpha,
                                 f"Noise Alpha Temporal Subject {data_sbj}, Noise: {noises_alpha[sbj]}", data_sbj)
        eeg_obj.plot_time_series(noise_delta,
                                 f"Noise Delta Temporal Subject {data_sbj}, Noise: {noises_delta[sbj]}", data_sbj)
        eeg_obj.plot_freq_resp(eeg_alpha, eeg_obj.fs,
                               f"Noisy Alpha Frequency Subject {data_sbj}, Noise: {noises_alpha[sbj]}", data_sbj)
        eeg_obj.plot_freq_resp(eeg_delta, eeg_obj.fs,
                               f"Noisy Delta Frequency Subject {data_sbj}, Noise: {noises_delta[sbj]}", data_sbj)
        eeg_obj.plot_freq_resp(noise_alpha, eeg_obj.fs,
                               f"Noise Alpha Frequency Subject {data_sbj}, Noise: {noises_alpha[sbj]}", data_sbj)
        eeg_obj.plot_freq_resp(noise_delta, eeg_obj.fs,
                               f"Noise Delta Frequency Subject {data_sbj}, Noise: {noises_delta[sbj]}", data_sbj)
        eeg_obj.plot_freq_resp_vs(0, pure_eeg_alpha, eeg_alpha, eeg_obj.fs,
                                  f"Pure vs Noisy Alphas Frequency Subject {data_sbj}, Noise: {noises_alpha[sbj]}", "Pure Alpha", "Noisy Alpha", data_sbj)
        eeg_obj.plot_freq_resp_vs(1, pure_eeg_delta, eeg_delta, eeg_obj.fs,
                                  f"Pure vs Noisy Deltas Frequency Subject {data_sbj}, Noise: {noises_delta[sbj]}", "Pure Delta", "Noisy Delta", data_sbj)
        # plt.show()
        noise_delta *= NOISE_GAIN_DELTA
        noise_alpha *= NOISE_GAIN_ALPHA
        eeg_alpha *= ALPHA_GAIN
        eeg_delta *= DELTA_GAIN
        print(
            f"Pure Alpha: {pure_alpha_SNR} dB")
        print(
            f"Pure Delta: {pure_delta_SNR} dB")
        eeg_obj.save_data(pure_eeg_alpha, noise_alpha, str(sbj), "Alpha/Pure")
        eeg_obj.save_data(eeg_alpha, noise_alpha, str(sbj), "Alpha/Noisy")
        eeg_obj.save_data(pure_eeg_delta, noise_delta, str(sbj), "Delta/Pure")
        eeg_obj.save_data(eeg_delta, noise_delta, str(sbj), "Delta/Noisy")
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


def get_results(signal):
    eeg_obj = EEG_GEN()
    results_df = pd.read_excel(
        f"Results-Generation/subj{data_sbj}/results_{signal}.xlsx", "DNF Results")
    init_df = pd.read_excel(r"initial_params.xlsx", signal)
    dnf_snrs = []
    lms_snrs = []
    laplace_snrs = []
    impulse = 0
    for sbj in range(noises):
        params_sbj = init_df.loc[init_df['Subject Number'] == sbj]
        dnf_data = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject_{signal}{sbj}.tsv")
        NOISE_GAIN = params_sbj["Noise Gain"].values
        SIG_GAIN = params_sbj["Signal Gain"].values
        dnf_pure = dnf_data[:, 0]
        dnf_noisy = dnf_data[:, 1]
        dnf_noisy = dnf_noisy[impulse:]
        dnf_noisy /= SIG_GAIN
        lms_data = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject_{signal}{sbj}.tsv")
        laplace_data = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/laplace_subject_{signal}{sbj}.tsv")
        lms_pure = lms_data[:, 0]
        lms_noisy = lms_data[:, 1]
        laplace_noisy = laplace_data[:, 1]
        laplace_noisy /= SIG_GAIN
        lms_noisy = lms_noisy[impulse:]
        lms_noisy /= SIG_GAIN
        remover_data = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject_{signal}{sbj}.tsv")
        remover_pure = remover_data[:, 0]
        remover_noisy = remover_data[:, 1]
        pure_sig = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/{signal}/Pure/EEG_Subject{sbj}.tsv")
        noisy_sig = np.loadtxt(
            f"deepNeuronalFilter/SubjectData/{signal}/Noisy/EEG_Subject{sbj}.tsv")
        inner_pure = pure_sig[:, 1]
        outer_pure = pure_sig[:, 2]
        inner_noisy = noisy_sig[:, 1]
        inner_noisy /= SIG_GAIN
        outer_noisy = noisy_sig[:, 2]
        inner_pure = inner_pure[impulse:]
        parameters = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/cppParams_subject_{signal}{sbj}.tsv")
        errors = np.loadtxt(
            f"deepNeuronalFilter/cppData/subject{sbj}/error_subject_{signal}{sbj}.tsv")
        dnf_error = errors[:, 0]
        lms_error = errors[:, 1]
        laplace_error = errors[:, 2]
        params = parameters[:]
        dnf_noisy = eeg_obj.butter_highpass(dnf_noisy, 1, eeg_obj.fs, 4)
        dnf_pure = eeg_obj.butter_highpass(dnf_pure, 1, eeg_obj.fs, 4)
        snr_fct = 1
        calc = 0
        if("Alpha" in signal):
            sig_type = 0
        elif("Delta" in signal):
            sig_type = 1
        else:
            print("Please enter the correct signal names!")
            exit(0)
        SNR_ORIG = params_sbj["Pure SNR"].values
        SNR_DNF = 10*np.log10(eeg_obj.calc_SNR(
            inner_pure, dnf_noisy, sig_type, snr_fct, calc))
        diff_dnf = SNR_DNF - SNR_ORIG[0]
        dnf_snrs.append(diff_dnf)
        SNR_LMS = 10*np.log10(eeg_obj.calc_SNR(
            inner_pure, lms_noisy, sig_type, snr_fct, calc))
        diff_lms = SNR_LMS - SNR_ORIG[0]
        lms_snrs.append(diff_lms)
        SNR_LAPLACE = 10*np.log10(eeg_obj.calc_SNR(
            inner_pure, laplace_noisy, sig_type, snr_fct, calc))
        diff_laplace = SNR_LAPLACE - SNR_ORIG[0]
        laplace_snrs.append(diff_laplace)
        print(f"{signal} Results: ")
        print(SNR_ORIG)
        print(SNR_DNF)
        print(SNR_LMS)
        print(SNR_LAPLACE)
        eeg_obj.plot_time_series(inner_pure[5000:6000],
                                 f"Pure {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)
        eeg_obj.plot_time_series(inner_noisy[5000:6000],
                                 f"Noisy {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)
        eeg_obj.plot_time_series(dnf_noisy[5000:6000],
                                 f"DNF {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)
        eeg_obj.plot_time_series(lms_noisy[5000:6000],
                                 f"LMS {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)
        eeg_obj.plot_time_series(outer_noisy,
                                 f"Noise Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)
        eeg_obj.plot_time_series(remover_noisy,
                                 f"Remover {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj)

        eeg_obj.plot_freq_resp(inner_noisy, Fs=eeg_obj.fs,
                               title=f"Noisy {signal} Frequency Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_subj=data_sbj)
        eeg_obj.plot_freq_resp_vs(sig_type,
                                  inner_noisy, dnf_noisy, eeg_obj.fs, f"Noisy vs Denoised {signal} Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", "Noisy {signal}", "Denoised {signal}", data_sbj)
        eeg_obj.plot_freq_resp_vs(sig_type,
                                  inner_pure, dnf_noisy, eeg_obj.fs, f"Pure vs Denoised {signal} Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", "Pure {signal}", "Denoised {signal}", data_sbj)
        eeg_obj.plot_freq_resp_vs(sig_type,
                                  inner_noisy, lms_noisy, eeg_obj.fs, f"Noisy vs Denoised {signal} LMS Subject {data_sbj}", "Noisy {signal}", "Denoised {signal}", data_sbj)
        eeg_obj.plot_freq_resp(outer_noisy, Fs=eeg_obj.fs,
                               title=f"Noise Frequency Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_subj=data_sbj)
        eeg_obj.plot_freq_resp(lms_noisy, Fs=eeg_obj.fs,
                               title=f"LMS {signal} Frequency Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_subj=data_sbj)
        eeg_obj.plot_freq_resp(remover_noisy, Fs=eeg_obj.fs,
                               title=f"Remover {signal} Frequency Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_subj=data_sbj)
        eeg_obj.plot_time_series(dnf_error[5000:6000],
                                 f"Error rates {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj, False)
        eeg_obj.plot_time_series(lms_error[5000:6000],
                                 f"Error rates {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj, False)
        eeg_obj.plot_time_series(laplace_error[5000:6000],
                                 f"Error rates {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}", data_sbj, False, True)
        plt.savefig(
            f"Results-Generation/subj{data_sbj}/Temporal_Error rates {signal} Temporal Subject {data_sbj}, Noise: {params_sbj['Noise'].values}")
        params = np.append(
            params, params_sbj["Amplitude Sig"])
        params = np.append(
            params, params_sbj["Frequency Sig"])
        params = np.append(
            params, params_sbj["Sum Sinusoids"])
        params = np.append(
            params, params_sbj["Bandpass Sinusoids"])
        params = np.append(
            params, params_sbj["Optimal Sinusoids"])
        params = np.append(params, SIG_GAIN)
        params = np.append(params, NOISE_GAIN)
        params = np.append(params, params_sbj["Pure SNR"])
        params = np.append(params, SNR_DNF)
        params = np.append(params, SNR_LMS)
        params = np.append(params, sbj)
        params = np.append(params, params_sbj["Noise"])
        results_df = results_df.append(pd.DataFrame(
            [params], columns=results_df.columns), ignore_index=True)
        writer = pd.ExcelWriter(
            f"Results-Generation/subj{data_sbj}/results_{signal}.xlsx", engine='xlsxwriter')
        results_df.to_excel(writer, "DNF Results", index=False)
        writer.save()
        writer.close()
    if(signal == "Alpha"):
        labels = noises_alpha
    elif(signal == "Delta"):
        labels = noises_delta
    labels = labels[:noises]
    dnf_snrs = dnf_snrs[:noises]
    lms_snrs = lms_snrs[:noises]
    laplace_snrs = laplace_snrs[:noises]
    bar_plt = eeg_obj.bar_plot(
        labels, dnf_snrs, lms_snrs, laplace_snrs, signal, data_sbj)


# gen_eeg()
with open('deepNeuronalFilter/signal.txt', 'r') as file:
    signal = file.read().replace('\n', '')
get_results(signal)
