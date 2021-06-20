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

pi = np.pi
FILT_ORDER = 3
MAX_AMP = 1e4  # max amp
SNR_d = 10  # dB
N = 2000  # Samples

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


def butter_bandpass(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


def end(x): return len(x) - 1


def brute_search(signal, target, params, tolerance, orders, diff=1e3):
    while(diff >= tolerance):

        pass


def to_polar(complex_ar):
    return np.abs(complex_ar), np.angle(complex_ar)

# ******************** Main ********************


orders = np.arange(1, 10, 1)
samples = np.arange(1e3, 9e3, 1e3)
amps = np.arange(0, MAX_AMP/4)
freq = np.arange(1, 200)
mat_00 = loadmat('Data/subject_00.mat')
mat_data_00 = mat_00["SIGNAL"]
signal_00 = mat_data_00[:, 1:17]
t = mat_data_00[:, 0][:N]
dt = t[1] - t[0]
fs = int(1/dt)  # Sampling frequency
alpha_00 = signal_00[:, 0][:N]  # Alpha wave
# Band Pass Filter between 8-12 Hz
filt_alpha_00 = butter_bandpass(alpha_00, 8, 12, fs, order=FILT_ORDER)
nfft = len(filt_alpha_00)
fft_alpha_00 = np.fft.fft(filt_alpha_00)  # FFT of signal
alpha_amp_00 = 2*np.abs(fft_alpha_00)/N  # Amplitude of signal
# Frequencies on axis in Hz of signal
alpha_hz_00 = np.linspace(0, fs/2, floor(N/2) + 1)
alpha_phi_00 = 2*np.angle(fft_alpha_00)/N  # Phase of EEG Alpha wave
# %%
plt.plot(t, filt_alpha_00)
plt.title("Real EEG (SUB_00)")
plt.ylabel("Voltage (uV)")
plt.xlabel("Time (s)")
plt.figure()
plt.plot(alpha_hz_00, alpha_amp_00[:len(alpha_hz_00)])
plt.xlabel("Frequencies (Hz)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Frequency Spectrum of Alpha EEG wave")
plt.figure()
win = 4 * fs
freqs, psd_2 = sig.welch(filt_alpha_00, fs, nperseg=win)
plt.plot(freqs, psd_2, color='k', lw=2)
plt.title("Real EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
# %%
f_start = 50
f_end = 60
f_tot = np.linspace(f_start, f_end, N)
f_final = np.cumsum(f_tot/fs)
A = MAX_AMP*np.exp(-t)
f = 7
A = A * np.sin(2*np.pi*f/fs)
phi = 0
eeg_sin = EEG(A, np.sin, pi, f_final, t, phi)
eeg_cos = EEG(A, np.cos, pi, f_final, t, phi)
# %%
f_sin = 10
sines = 1.2 * np.sin(2 * pi * f_sin * t + 1.2) + 1.4 * np.sin(2 *
                                                              pi * f_sin * t + 0.3) + 0.9 * np.sin(2 * pi * f_sin * t - 0.45)
# Add AWGN based on desired SNR
RMS_s = sqrt(np.mean(eeg_sin.signal)**2)
RMS_n = sqrt(RMS_s**2/pow(10, SNR_d/10))
STD_n = RMS_n
AWGN = np.random.normal(0, STD_n, eeg_sin.signal.shape[0])
X = np.fft.rfft(AWGN)
radius, angle = to_polar(X)
plt.plot(radius)
plt.title("AWGN FR")
plt.xlabel("FFT coefficient")
plt.ylabel("Magnitude")
plt.figure()
noise_amp = 5
gauss_noise = noise_amp*np.random.normal(0, 1, len(eeg_sin.signal))
fake_eeg = (eeg_sin.signal) + sines + AWGN
low_cut = 8
high_cut = 12
eeg_dat = np.fromfile('subj27/blink.dat')
noise_freq = 50  # Hz
eeg_noise = eeg_dat[:len(fake_eeg)]
eeg_noise_fft = np.fft.fft(eeg_noise)
eeg_noise_amp = 2*np.abs(eeg_noise_fft)/N
yn = eeg_noise_amp*np.sin(2*pi*noise_freq*t)
noisy_eeg = fake_eeg + yn
filt_eeg = butter_bandpass(noisy_eeg, low_cut, high_cut, fs, order=FILT_ORDER)
nfft = len(filt_eeg)
fft_eeg = np.fft.fft(filt_eeg)  # FFT of signal
eeg_amp = 2*np.abs(filt_eeg)/N  # Amplitude of signal
# Frequencies on axis in Hz of signal
eeg_hz = np.linspace(0, fs/2, floor(N/2) + 1)
plt.plot(t, filt_eeg)
plt.title("Fake EEG")
plt.ylabel("Signal Voltage (V)")
plt.xlabel("Time (s)")
plt.figure()
plt.plot(eeg_hz, eeg_amp[:len(eeg_hz)])
plt.xlabel("Frequencies (Hz)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Frequency Spectrum of Alpha EEG wave")
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(filt_eeg, fs, nperseg=win)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Fake EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
distance, path = fastdtw(psd_1, psd_2, dist=euclidean)
print(f"Difference btw signals: {distance}")
# %%
# TODO: Compare with other EEG signals.
mat_01 = loadmat('Data/subject_01.mat')
mat_data_01 = mat_01["SIGNAL"]
signal_01 = mat_data_01[:, 1:17]
alpha_01 = signal_01[:, 0][:N]  # Alpha wave
# Band Pass Filter between 8-12 Hz
filt_alpha_01 = butter_bandpass(alpha_01, 8, 12, fs, order=FILT_ORDER)
plt.plot(t, filt_alpha_01)
plt.title("Real EEG (SUB_01)")
plt.ylabel("Voltage (uV)")
plt.xlabel("Time (s)")
plt.figure()
win = 4 * fs
freqs, psd = sig.welch(filt_alpha_01, fs, nperseg=win)
plt.plot(freqs, psd, color='k', lw=2)
plt.title("Real EEG FR (SUB_01)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.show()
# for i in range(end(filt_eeg)):
#     plt.scatter(t[i], filt_alpha_01[i], c="r")
#     plt.scatter(t[i], filt_eeg[i], c="b")
#     plt.pause(0.0001)
# plt.show()
