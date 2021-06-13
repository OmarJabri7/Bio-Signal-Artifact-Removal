import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
from eeg import EEG
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from math import floor
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

pi = np.pi
FILT_ORDER = 2


def butter_bandpass(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


def end(x): return len(x) - 1


# ******************** Main ********************
mat = loadmat('Data/subject_00.mat')
mat_data = mat["SIGNAL"]
signal = mat_data[:, 1:17]
N = 2000  # Samples
t = mat_data[:, 0][:N]
dt = t[1] - t[0]
fs = int(1/dt)  # Sampling frequency
alpha = signal[:, 0][:N]  # Alpha wave
# Band Pass Filter between 8-12 Hz
filt_alpha = butter_bandpass(alpha, 8, 12, fs, order=FILT_ORDER)
nfft = len(filt_alpha)
fft_alpha = np.fft.fft(filt_alpha)  # FFT of signal
alpha_amp = 2*np.abs(fft_alpha)/N  # Amplitude of signal
# Frequencies on axis in Hz of signal
alpha_hz = np.linspace(0, fs/2, floor(N/2) + 1)
alpha_phi = 2*np.angle(fft_alpha)/N  # Phase of EEG Alpha wave
# %%
plt.plot(t, filt_alpha)
plt.title("Real EEG (FP1)")
plt.ylabel("Voltage (uV)")
plt.xlabel("Time (s)")
plt.figure()
plt.plot(alpha_hz, alpha_amp[:len(alpha_hz)])
plt.xlabel("Frequencies (Hz)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Frequency Spectrum of Alpha EEG wave")
plt.figure()
win = 4 * fs
freqs, psd_2 = sig.welch(filt_alpha, fs, nperseg=win)
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
MAX_AMP = 1e4  # max amp
A = MAX_AMP*np.exp(-t)
f = 7
A = A * np.sin(2*np.pi*f/fs)
phi = 0
eeg_sig = EEG(A, np.sin, pi, f_final, t, phi)
eeg_cos = EEG(A, np.cos, pi, f_final, t, phi)
# %%
f_sin = 10
sines = 1.2 * np.sin(2 * pi * f_sin * t + 1.2) + 1.4 * np.sin(2 *
                                                              pi * f_sin * t + 0.3) + 0.9 * np.sin(2 * pi * f_sin * t - 0.45)
noise_amp = 5
gauss_noise = noise_amp*np.random.normal(0, 1, len(eeg_sig.signal))
fake_eeg = (eeg_sig.signal) + sines + gauss_noise
low_cut = 8
high_cut = 12
eeg_dat = np.fromfile('subj27/blink.dat')
filt_eeg = fake_eeg + eeg_dat[:len(fake_eeg)]
filt_eeg = butter_bandpass(filt_eeg, low_cut, high_cut, fs, order=FILT_ORDER)
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
plt.show()
distance, path = fastdtw(psd_1, psd_2, dist=euclidean)
print(f"Difference btw signals: {distance}")
# %%
# TODO: Compare with other EEG signals.
# eeg_data = pd.read_csv('EEG_sample.csv')
# sensors = np.unique(eeg_data["sensor position"])
# FP1 = eeg_data.loc[eeg_data['sensor position'] == "F7"]
# fp1_eeg = FP1["sensor value"]
# fp1_time = FP1["time"]
# plt.plot(fp1_time, fp1_eeg)
# plt.title("Real EEG (FP1)")
# plt.ylabel("Voltage (uV)")
# plt.xlabel("Time (s)")
# plt.figure()
# win = 4 * fs
# freqs, psd = sig.welch(fp1_eeg, fs, nperseg=win)
# plt.plot(freqs, psd, color='k', lw=2)
# plt.title("Real EEG FR")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectral density (V^2 / Hz)')
# plt.show()
