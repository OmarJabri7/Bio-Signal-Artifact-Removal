import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from eeg import EEG
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from math import floor, sqrt, pow
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.animation as anim
from scipy import stats

pi = np.pi

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


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
    SNR = 20 * np.log10(sig_var / noise_var)
    return SNR


def butter_bandpass(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
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
    plt.figure()


def plot_welch(signal, Fs, title, plot):
    win = 4 * Fs
    freqs, psd = sig.welch(signal, Fs, nperseg=win)
    if(plot == True):
        plt.plot(freqs, psd, color='k', lw=2)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.figure()
    return psd


def plot_time_series(signal, title):
    plt.plot(signal)
    plt.title(title)
    plt.ylabel("Signal Voltage (V)")
    plt.xlabel("Time (s)")
    plt.figure()


# def read_EEG_pure(filename):
#     a = np.loadtxt(filename)
#     f = a[:, 0]
#     p = a[:, 1]
#     psd = np.interp1d(f, p, kind='cubic')
#     bandpower = 0
#     for f2 in np.arange(f_signal_min, f_signal_max):
#         bandpower = bandpower + (10**psd(f2)) * \
#             (eegFilterFrequencyResponse[f2]**2)
#     return bandpower
    # ******************** Main ********************

    # %%
print("Show Plots: (y|n)")
show_plots = input().lower()
sbj = 1
eeg_dat = np.loadtxt('Data/subj27/word_search.dat')
blink_dat = np.loadtxt('Data/subj27/blink.dat')
relax_dat = np.loadtxt('Data/subj27/sit_relax.dat')
N = eeg_dat.shape[0]  # Samples
ts = eeg_dat[:, 0]
noise_fs = 1/(ts[1] - ts[0])
Fs = 1000.0
f = 10
sample = N
x = np.arange(sample)
A = 0.1
y = A*np.sin(2 * np.pi * f * x / Fs)
# %%
# Add AWGN based on desired SNR
N = min(eeg_dat.shape[0], blink_dat.shape[0])
y = y[:N]
eeg_noise = eeg_dat[:N, 1]
eeg_noise_2 = eeg_dat[:N, 2]
blink_noise = blink_dat[:N, 1]
blink_noise_2 = blink_dat[:N, 2]
relax_noise = relax_dat[:N, 1]
relax_noise = relax_dat[:N, 2]
noise = relax_noise
fake_eeg = y + noise
# eeg_pow = plot_welch(fake_eeg, Fs, "", False)
eeg_pow = np.abs(np.fft.fft(fake_eeg))**2
Ps = 1/N * (np.sum(fake_eeg**2))
# noise_pow = plot_welch(noise, Fs, "", False)
noise_pow = np.abs(np.fft.fft(noise))**2
Pn = 1/N*(np.sum(noise**2))
noise_var = np.var(noise)
sig_var = np.var(fake_eeg)
gain = 0.1
fake_eeg *= gain
noise *= gain
# SNR = signaltonoise(fake_eeg)
# SNR = calc_SNR(noise_var, sig_var)
SNR = 20*np.log10(np.mean(eeg_pow/noise_pow))
# SNR = 10 * np.log10(np.abs(Ps - Pn)/Pn)
print(f"SNR: {SNR} dB")
eyes_closed = np.loadtxt('Data/Novel_Subject.tsv')
eyes_closed_inner = eyes_closed[:, 2]
eyes_closed_outer = eyes_closed[:, 2]
zeros = np.zeros(len(fake_eeg))
sig_fake = np.column_stack((fake_eeg, noise, zeros))
signal_df = pd.DataFrame(sig_fake)
signal_df.to_csv(f"deepNeuronalFilter/SubjectData/EEG_Subject{sbj}.tsv",
                 index=True, header=False,  sep="\t")
dnf_res = pd.read_csv(
    f"deepNeuronalFilter/cppData/subject{sbj}/fnn_subject{sbj}.tsv", sep=" ")
res = dnf_res.values
init = 0
res = res[init:N + init, 0]
res_var = np.var(res)
# res_pow = plot_welch(res, Fs, "", False)
res_pow = np.abs(np.fft.fft(res))**2
# Pr = 1/N*(np.sum(res**2))
# SNR_NEW = signaltonoise(res)
# SNR_NEW = calc_SNR(noise_var, res_var)
SNR_NEW = 20*np.log10(np.mean(res_pow/noise_pow))
# SNR_NEW = 10*np.log10(np.abs(Pr - Pn)/Pn)
print(f"SNR NEW: {SNR_NEW} dB")
plot_time_series(fake_eeg, "Fake EEG")
plot_time_series(noise, "Noise")
plot_time_series(res, "DNF Result")
plot_time_series(eyes_closed_inner, "Real EEG")
plot_freq_resp(fake_eeg, Fs, "Fake EEG")
plot_freq_resp(res, Fs, "DNF result")
plot_freq_resp(eyes_closed_inner, Fs, "Real EEG")
if(show_plots == "y"):
    plt.show()
