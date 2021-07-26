from noise_source import EEG_Noise
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from param_search import fit_sin
from scipy.io import loadmat
from matplotlib.backends.backend_pdf import PdfPages
pi = np.pi
# plt.ion()

# use ggplot style for more sophisticated visuals
# plt.style.use('ggplot')


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=2):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),
                 np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1


def calc_SNR(noise_var, sig_var):
    SNR = 10 * np.log10(sig_var / noise_var)
    return SNR


def get_SNR(signal, outer, inner, remover, fnn, f_low, f_high, N, fs=250):
    timeX = np.linspace(0, N / 250, N)
    f_cut_start = int((f_low / fs) * N)
    f_cut_end = int((f_high / fs) * N)
    freqX = np.linspace(0, fs / 2, int(N / 2))
    fft_outer = np.abs(np.fft.fft(outer)[
        0:np.int(N / 2)])
    fft_inner = np.abs(np.fft.fft(inner)[
        0:np.int(N / 2)])
    # fft_fnn = np.abs(np.fft.fft(fnn)[
    # 0:np.int(N / 2)])
    # df_inner = pd.DataFrame({"x": freqX, "y": fft_inner})
    # # df_fnn = pd.DataFrame({"x": freqX, "y": fft_fnn})
    sin_start = int((f_low / fs) * N)
    sin_end = int((f_high / fs) * N)
    diff_noise = sin_end - sin_start
    noise = np.sum(
        fft_outer[sin_start:sin_end]) / diff_noise
    signal_integral = np.sum(
        fft_inner) / diff_noise
    signal = signal_integral - noise
    snr = signal / noise
    return np.sum(snr)


def butter_bandpass(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


def butter_lowpass(signal, cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, signal)
    return y


def end(x): return len(x) - 1


def to_polar(complex_ar):
    return np.abs(complex_ar), np.angle(complex_ar)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


def plot_freq_resp(signal, Fs, title):
    fourierTransform = np.fft.fft(signal)/len(signal)
    fourierTransform = fourierTransform[range(int(len(signal)/2))]
    tpCount = len(signal)
    values = np.arange(int(tpCount/2))
    timePeriod = tpCount/Fs
    frequencies = values/timePeriod
    plt.plot(frequencies, abs(fourierTransform))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.savefig(f"Results-Generation/Frequency_{title}")
    fig = plt.figure()
    return fig


def plot_welch(signal, Fs, title):
    win = 10*Fs
    freqs, psd = sig.welch(signal, Fs, nperseg=win)
    plt.plot(freqs, psd, color='k', lw=2)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    fig = plt.figure()
    return fig


def plot_time_series(signal, title):
    plt.plot(signal)
    plt.title(title)
    plt.ylabel("Signal Voltage (V)")
    plt.xlabel("Time (s)")
    plt.savefig(f"Results-Generation/Temporal_{title}")
    fig = plt.figure()
    return fig


def filter_banks(signal, fs, nfilt, NFFT):
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_stride = 0.01
    frame_size = 0.025
    frame_length, frame_step = frame_size * fs, frame_stride * \
        fs  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(
        frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / fs)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(
        float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
# ******************** Main ********************


# %%
print("Show Plots: (y|n)")
show_plots = input().lower()
print("Save signals: (y|n)")
save_sigs = input().lower()
print("Update Csv: (y|n)")
updt_csv = input().lower()
sbjcts = 1
sudoku_dat = np.loadtxt('Data/subj27/sudoku.dat')
blink_dat = np.loadtxt('Data/subj27/blink.dat')
relax_dat = np.loadtxt('Data/subj27/sit_relax.dat')
ts = sudoku_dat[:, 0]
noise_fs = 1/(ts[1] - ts[0])
Fs = 1000.0
f = 10
N = min(sudoku_dat.shape[0], blink_dat.shape[0], relax_dat.shape[0])
x = np.arange(N)
A = 0.1
y = A*np.sin(2 * np.pi * f * x / Fs)
y = y[:N]
# Add noise based on desired SNR
sudoku_noise = sudoku_dat[:N, 1]
sudoku_noise_2 = sudoku_dat[:N, 2]
blink_noise = blink_dat[:N, 1]
blink_noise_2 = blink_dat[:N, 2]
relax_noise = relax_dat[:N, 1]
relax_noise_2 = relax_dat[:N, 2]
noise = [relax_noise_2, blink_noise]
# Setup SNR parameters for integral calculations
Py = 1/N*(np.sum(y**2))
# DC Removal
# y -= np.mean(y)
if(not os.path.exists("results.xlsx")):
    data_frame = pd.DataFrame(
        columns=["Layers Number", "Outer Inputs", "Inner Inputs", "Outer Gain", "Inner Gain", "Remover Gain", "Feedback Gain", "weight Eta", "bias Eta",  "Signal Gain", "Noise Gain", "SNR Before DNF", "SNR After DNF"])
    data_frame.to_csv("results.xlsx", index=False, sep=",")
results_df = pd.read_excel("results.xlsx", "DNF results")
# %%
for sbj in range(sbjcts):
    print(f"Subject {sbj}: ")
    gain_eeg = 0.2
    gain_noise = 0.4
    # y *= gain_eeg
    # noise[sbj] *= gain_noise
    fake_eeg = y + noise[sbj]
    fake_eeg = fake_eeg - np.mean(fake_eeg, axis=-1)
    fake_eeg *= gain_eeg
    noise[sbj] *= gain_noise
    Ps = 1/N * (np.sum(fake_eeg**2))
    Pn = 1/N * (np.sum(noise[sbj]**2))
    SNR = 10 * np.log10(Ps/(Pn))
    print(f"SNR Before: {SNR}")
    zeros = np.zeros(len(fake_eeg))
    sig_fake = np.column_stack((fake_eeg, noise[sbj]))
    signal_df = pd.DataFrame(sig_fake)
    if(save_sigs == "y"):
        signal_df.to_csv(
            f"deepNeuronalFilter/SubjectData/EEG_Subject{sbj}.tsv", index=True, header=False, sep="\t")
        print("Signals Saved!")
    dnf_res = np.loadtxt(
        f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject{sbj}.tsv")
    lms_res = np.loadtxt(
        f"deepNeuronalFilter/cppData/subject{sbj}/lmsOutput_subject{sbj}.tsv")
    remover_res = np.loadtxt(
        f"deepNeuronalFilter/cppData/subject{sbj}/remover_subject{sbj}.tsv")
    parameters_res = np.loadtxt(
        f"deepNeuronalFilter/cppData/subject{sbj}/cppParams_subject{sbj}.tsv")
    init = 0
    dnf_res = dnf_res[init:N + init]
    Pr = 1/N*(np.sum(dnf_res**2))
    Prem = 1/N*(np.sum(remover_res**2))
    Plms = 1/N*(np.sum(lms_res**2))
    SNR_DNF = 10*np.log10(Pr/(Pn))
    SNR_REM = 10*np.log10((Prem)/Pn)
    SNR_LMS = 10*np.log10((Plms) / (Pn))
    print(f"SNR After: {SNR_DNF}")
    parameters_res = np.append(parameters_res, gain_eeg)
    parameters_res = np.append(parameters_res, gain_noise)
    parameters_res = np.append(parameters_res, SNR)
    parameters_res = np.append(parameters_res, SNR_DNF)
    results_df = results_df.append(pd.DataFrame(
        [parameters_res], columns=results_df.columns), ignore_index=True)
    if(updt_csv == "y"):
        results_df.to_excel("results.xlsx", "DNF results", index=False)
    if(show_plots == "y"):
        fr1 = plot_welch(fake_eeg, Fs, "Fake EEG")
        fr2 = plot_welch(dnf_res, Fs, "DNF result")
        fr3 = plot_welch(remover_res, Fs, "Remover Result")
        fr4 = plot_welch(lms_res, Fs, "LMS Result")
        multipage("Plots.pdf")
