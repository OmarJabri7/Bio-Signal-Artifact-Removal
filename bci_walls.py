import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import scipy.signal as sig
import scipy


def butter_bandpass(data, Wn, fs, order=5):
    nyq = 0.5 * fs
    Wn = Wn/nyq
    print(nyq)
    print(Wn)
    sos = sig.butter(order, Wn, analog=False, btype='band', output='sos')
    y = sig.sosfilt(sos, data)
    return y


eeg_data = pd.read_csv('EEG_sample.csv')

sensors = np.unique(eeg_data["sensor position"])

F1 = eeg_data.loc[eeg_data['sensor position'] == "F1"]

eeg = F1["sensor value"]

eeg = np.array(eeg).reshape(-1, 1)

t = np.arange(len(eeg))

plt.plot(t, eeg)

plt.xlabel("Trials")
plt.ylabel("Gain")
plt.figure()

fs_hz = 256  # Hz

fs_rad = 256*6.28

w, h = sig.freqz(eeg)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [rad/sample]')
plt.figure()

# pass band frequency
Wn = np.array([0.1, 1.0])

filt_eeg = butter_bandpass(eeg, Wn, fs_rad)

w, h = sig.freqz(filt_eeg)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [rad/sample]')

plt.figure()

plt.plot(filt_eeg)

plt.xlabel("Trials")
plt.ylabel("Gain")
plt.figure()

plt.show()
