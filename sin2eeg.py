import numpy as np
import matplotlib.pyplot as plt
from numpy.core.arrayprint import format_float_positional
import scipy.signal as sig
import pandas as pd
from eeg import EEG
from scipy.signal import butter, lfilter

pi = np.pi


def butter_bandpass(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


# ******************** Main ********************
fs = 250
T = 1/fs
N = fs*4
t = np.arange(N)*T
MAX_AMP = 5000  # max amp
A = MAX_AMP*np.exp(-0.0003*t)
A = A * np.sin(2*np.pi*t/2000)
f_start = 0
f_end = 60
f_tot = np.linspace(f_start, f_end, len(t))
f_final = np.cumsum(f_tot/fs)
phi = 0.0
f = 8
eeg_sig = EEG(A, np.sin, pi, f_final, t, phi)
eeg_cos = EEG(A, np.cos, pi, f_final, t, phi)
# %%
noise = 1.2 * np.sin(2 * pi * 5.36 * t + 1.2) + 1.4 * np.sin(2 *
                                                             pi * 7.34 * t + 0.3) + 0.9 * np.sin(2 * pi * 9.24 * t - 0.45) + 2.1 * np.sin(2 * pi * 5.4 * t - 1.4)
noise_amp = 5
gauss_noise = noise_amp*np.random.normal(0, 1, len(eeg_sig.signal))
fake_eeg = ((eeg_sig.signal) * noise) + gauss_noise
low_cut = 8
high_cut = 12
filt_eeg = butter_bandpass(fake_eeg, low_cut, high_cut, fs)
eeg_dat = np.fromfile('subj27/blink.dat')
filt_eeg = filt_eeg + eeg_dat[:len(filt_eeg)]
plt.plot(t, filt_eeg)
plt.title("Fake EEG")
plt.ylabel("Voltage (uV)")
plt.xlabel("Time (s)")
plt.figure()
win = 4 * fs
freqs, psd = sig.welch(filt_eeg, fs, nperseg=win)
plt.plot(freqs, psd, color='k', lw=2)
plt.title("Fake EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
# # %%
eeg_data = pd.read_csv('EEG_sample.csv')
sensors = np.unique(eeg_data["sensor position"])
FP1 = eeg_data.loc[eeg_data['sensor position'] == "F1"]
fp1_eeg = FP1["sensor value"]
fp1_time = FP1["time"]
plt.plot(fp1_time, fp1_eeg)
plt.title("Real EEG (FP1)")
plt.ylabel("Voltage (uV)")
plt.xlabel("Time (s)")
plt.figure()
win = 4 * fs
freqs, psd = sig.welch(fp1_eeg, fs, nperseg=win)
plt.plot(freqs, psd, color='k', lw=2)
plt.title("Real EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.show()
