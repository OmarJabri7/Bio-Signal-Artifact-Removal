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

# ******************** Main ********************


# %%
sbj = 1
eeg_dat = np.loadtxt('Data/subj27/word_search.dat')
blink_dat = np.loadtxt('Data/subj27/blink.dat')
relax_dat = np.loadtxt('Data/subj27/sit_relax.dat')
N = eeg_dat.shape[0]  # Samples
ts = eeg_dat[:, 0]
noise_fs = 1/(ts[1] - ts[0])
print(noise_fs)
Fs = 1000.0
f = 10
sample = N
x = np.arange(sample)
A = 0.01
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
fake_eeg = y + relax_noise  # + blink_noise  # + eeg_noise_2 + \
# eeg_noise + blink_noise_2  # + sines + AWGN
fake_eeg *= A
blink_noise *= A
snr = signaltonoise(fake_eeg)
print(f"SNR: {snr}")
eyes_closed = np.loadtxt('Data/Novel_Subject.tsv')
eyes_closed_inner = eyes_closed[:, 2]
eyes_closed_outer = eyes_closed[:, 2]
zeros = np.zeros(len(fake_eeg))
sig_fake = np.column_stack((blink_noise, fake_eeg, zeros))
signal_df = pd.DataFrame(sig_fake)
signal_df.to_csv(f"deepNeuronalFilter/SubjectData/EEG_Subject{sbj}.tsv",
                 index=True, header=False,  sep="\t")
plt.plot(fake_eeg)
plt.title("Fake EEG")
plt.ylabel("Signal Voltage (V)")
plt.xlabel("Time (s)")
# plt.ylim(-10, 1)
plt.figure()
plt.plot(eyes_closed_inner)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Real EEG (DNF INNER)")
plt.figure()
win = 4 * Fs
freqs, psd_1 = sig.welch(fake_eeg, Fs, nperseg=win)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Fake EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
win = 4 * Fs
freqs, psd_1 = sig.welch(eyes_closed_inner, Fs, nperseg=win)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Real EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
dnf_res = pd.read_csv('Results-Generation/remover_subject1.tsv', sep=" ")
res = dnf_res.values
res = res[600:, :]
print("SNR NEW: " + str(signaltonoise(res)))
plt.plot(res[:, 0])
plt.ylim(-0.01, 0.01)
plt.show()
