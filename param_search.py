from numpy.lib.twodim_base import eye
from eeg import EEG
import scipy.optimize
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from math import floor, sqrt
from scipy.io import loadmat
from scipy.signal import chirp
from scipy.optimize import leastsq
from scipy.signal.ltisys import freqresp
import pandas as pd
plt.style.use('ggplot')
# ******************** Main ********************
SNR_d = 10  # dB


def butter_bandpass(data, Wn, fs, order=5):
    nyq = 0.5 * fs
    Wn = Wn/nyq
    sos = sig.butter(order, Wn, analog=False, btype='band', output='sos')
    y = sig.sosfilt(sos, data)
    return y


def autocorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def band_limited_noise(noise, noise_lvl, min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = noise
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(noise_lvl*f)


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    # excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c): return A * np.sin(w*t + p) + c

    popt, pcov = scipy.optimize.curve_fit(
        sinfunc, tt, yy, p0=guess, maxfev=5000)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    def fitfunc(t): return A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}


print("Enter subject no:")
subj_no = input()
noise_idx = 1
eeg_dat = np.loadtxt('Data/subj27/blink.dat')
sudoku_dat = np.loadtxt('Data/subj27/sudoku.dat')
read_dat = np.loadtxt('Data/subj27/lie_read.dat')
word_dat = np.loadtxt('Data/subj27/word_search.dat')
ecg_dat = np.loadtxt('Data/ecg_50hz_1.dat.txt')
pi = np.pi
mat = loadmat(f"Data/subject_{subj_no}.mat")
mat_data = mat["SIGNAL"]
signal = mat_data[:, 1:17]
N = min(signal.shape[0], eeg_dat.shape[0])
t = mat_data[:, 0][:N]
dt = t[1] - t[0]
fs = int(1/dt)  # Sampling frequency
eeg = signal[:, 0][:N]  # Alpha wave
# %%
signal_eeg = np.interp(eeg, (eeg.min(), eeg.max()), (-1, 1))
fft_eeg = np.fft.fft(signal_eeg)
mean_eeg = np.mean(signal_eeg)
std_eeg = np.std(signal_eeg)
amp_eeg = np.abs(fft_eeg/N)**2
phi_eeg = 2*np.angle(fft_eeg)/N
eeg_freqs = np.linspace(0, fs/2, floor(N/2) + 1)
sin2eeg = std_eeg*np.sin(t + phi_eeg) + mean_eeg
RMS_s = sqrt(np.mean(sin2eeg)**2)
RMS_n = sqrt(RMS_s**2/pow(10, SNR_d/10))
STD_n = RMS_n
noise_amp = 1
AWGN = np.random.normal(0, STD_n, sin2eeg.shape[0])

dic = fit_sin(t, signal_eeg)
sin2eeg = dic["amp"]*np.sin(dic["omega"]*t + dic["phase"]) + dic["offset"]
sin2eeg = sig.detrend(sin2eeg)
noise_freq = 50  # Hz
eeg_noise = eeg_dat[:N, 1]
eeg_noise_2 = eeg_dat[:N, 2]
sudoku_noise = sudoku_dat[:N, 1]
sudoku_noise_2 = sudoku_dat[:N, 2]
read_noise = read_dat[:N, 1]
read_noise_2 = read_dat[:N, 2]
word_noise = word_dat[:N, 1]
word_noise_2 = word_dat[:N, 2]
ecg_noise = ecg_dat[:N, 0]
ecg_noise_2 = ecg_dat[:N, 1]
f_sin = 8
sines = 1.2 * np.sin(2 * pi * f_sin * t + 1.2) + 1.4 * np.sin(2 *
                                                              pi * f_sin * t + 0.3)
noise_lvl = 1
# eeg_noise = band_limited_noise(eeg_noise, noise_lvl, 48, 52, N, fs)
Wn = np.array([8, 13])
fs_rad = fs*6.28
# sin2eeg = butter_bandpass(sin2eeg, Wn, fs_rad)
sin2eeg = sin2eeg + eeg_noise  # + eeg_noise_2 + \
# sudoku_noise + sudoku_noise_2 + read_noise + read_noise_2 + \
# word_noise + word_noise_2  # + ecg_noise + ecg_noise_2  # + AWGN
snr = signaltonoise(sin2eeg)
print(f"SNR: {snr}")
zeros = np.zeros(len(sin2eeg))
sig_eeg = np.column_stack((sin2eeg, sin2eeg, zeros))
sin_peaks = sig.find_peaks(sin2eeg)[0]
plt.plot(t, signal_eeg)
plt.plot(t, sin2eeg)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Fake VS Real EEG (Subject {subj_no})")
plt.legend(["Real EEG", "Sine EEG"])
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(sin2eeg, fs, nperseg=fs)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Fake EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(signal_eeg, fs, nperseg=fs)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Real EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
# Check Correlation
corr_sig = sig.correlate(sin2eeg, eeg)
print(f"Correlation: {np.mean(corr_sig)}")
# Save signal as dat file
signal_df = pd.DataFrame(sig_eeg)
signal_df.to_csv("Results-Generation/EyesClosedNovel_Subject1.csv",
                 index=True, header=False, sep="\t")
# Try other dataset
eyes_closed = np.loadtxt('Data/EyesClosedNovel_Subject1.tsv')
eyes_closed_inner = eyes_closed[:, 2]
eyes_closed_outer = eyes_closed[:, 2]
# Inner
N = min(eyes_closed_inner.shape[0], eeg_dat.shape[0])
fs = 250
T = 1/fs
t_in = np.arange(N)*T
eyes_closed_inner = eyes_closed_inner[:N]
dic_fp1 = fit_sin(t_in, eyes_closed_inner)
fft_eeg = np.fft.fft(eyes_closed_inner)
mean_eeg = np.mean(eyes_closed_inner)
std_eeg = np.std(eyes_closed_inner)
amp_eeg = np.abs(fft_eeg/N)**2
phi_eeg = 2*np.angle(fft_eeg)/N
eeg_freqs = np.linspace(0, fs/2, floor(N/2) + 1)
sin2fp1 = std_eeg*np.sin(t_in + phi_eeg) + mean_eeg
RMS_s = sqrt(np.mean(sin2fp1)**2)
RMS_n = sqrt(RMS_s**2/pow(10, SNR_d/10))
STD_n = RMS_n
noise_amp = 1
AWGN = np.random.normal(0, STD_n, sin2fp1.shape[0])
sin2fp1 = dic_fp1["amp"]*np.sin(dic_fp1["omega"]
                                * t_in + dic_fp1["phase"]) + dic_fp1["offset"]
noise_lvl = 0.01
eeg_noise = eeg_dat[:N, noise_idx]
# eeg_noise = band_limited_noise(eeg_noise, noise_lvl, 48, 52, N, fs)
noise_amp = 1
gauss_noise = noise_amp*np.random.normal(0, 1, len(sin2fp1))
eeg_noise = eeg_dat[:N, 1]
eeg_noise_2 = eeg_dat[:N, 2]
sudoku_noise = sudoku_dat[:N, 1]
sudoku_noise_2 = sudoku_dat[:N, 2]
read_noise = read_dat[:N, 1]
read_noise_2 = read_dat[:N, 2]
word_noise = word_dat[:N, 1]
word_noise_2 = word_dat[:N, 2]
ecg_noise = ecg_dat[:N, 0]
ecg_noise_2 = ecg_dat[:N, 1]
sin2fp1 = sin2fp1 + eeg_noise + eeg_noise_2 + sudoku_noise + \
    sudoku_noise_2 + read_noise + read_noise_2 + word_noise + \
    word_noise_2  # + ecg_noise + ecg_noise_2  # + gauss_noise
zeros = np.zeros(len(sin2fp1))
sig_eeg = np.column_stack((sin2fp1, sin2fp1, zeros))
signal_df = pd.DataFrame(sig_eeg)
signal_df.to_csv("Results-Generation/EyesClosedNovel_Subject2.csv",
                 index=True, header=False, sep="\t")
plt.plot(t_in, eyes_closed_inner)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Real EEG (DNF INNER)")
plt.figure()
plt.plot(t_in, sin2fp1)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Fake EEG (DNF INNER)")
plt.legend(["Real EEG", "Sine EEG"])
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(sin2fp1, fs, nperseg=fs)
plt.plot(freqs, psd_1, lw=2)
plt.title("Fake EEG FR (DNF INNER)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(eyes_closed_inner, fs, nperseg=fs)
plt.plot(freqs, psd_1, lw=2)
plt.title("Real EEG FR (DNF INNER)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
# Outer
N = min(eyes_closed_outer.shape[0], eeg_dat.shape[0])
fs = 250
T = 1/fs
t_out = np.arange(N)*T
eyes_closed_outer = eyes_closed_outer[:N]
dic_fp1 = fit_sin(t_out, eyes_closed_outer)
fft_eeg = np.fft.fft(eyes_closed_outer)
mean_eeg = np.mean(eyes_closed_outer)
std_eeg = np.std(eyes_closed_outer)
amp_eeg = np.abs(fft_eeg/N)**2
phi_eeg = 2*np.angle(fft_eeg)/N
eeg_freqs = np.linspace(0, fs/2, floor(N/2) + 1)
sin2outer = std_eeg*np.sin(t_out + phi_eeg) + mean_eeg
RMS_s = sqrt(np.mean(sin2outer)**2)
RMS_n = sqrt(RMS_s**2/pow(10, SNR_d/10))
STD_n = RMS_n
noise_amp = 1
AWGN = np.random.normal(0, STD_n, sin2outer.shape[0])
sin2outer = dic_fp1["amp"]*np.sin(dic_fp1["omega"]
                                  * t_out + dic_fp1["phase"]) + dic_fp1["offset"]
noise_lvl = 0.01
eeg_noise = eeg_dat[:N, 1]
eeg_noise_2 = eeg_dat[:N, 2]
sudoku_noise = sudoku_dat[:N, 1]
sudoku_noise_2 = sudoku_dat[:N, 2]
read_noise = read_dat[:N, 1]
read_noise_2 = read_dat[:N, 2]
word_noise = word_dat[:N, 1]
word_noise_2 = word_dat[:N, 2]
ecg_noise = ecg_dat[:N, 0]
ecg_noise_2 = ecg_dat[:N, 1]
# eeg_noise = band_limited_noise(eeg_noise, noise_lvl, 48, 52, N, fs)
noise_amp = 1
gauss_noise = noise_amp*np.random.normal(0, 1, len(sin2outer))
sin2outer = sin2outer + eeg_noise + eeg_noise_2 + sudoku_noise + \
    sudoku_noise_2 + read_noise + read_noise_2 + word_noise + \
    word_noise_2  # + ecg_noise + ecg_noise_2  # + gauss_noise
zeros = np.zeros(len(sin2outer))
sig_eeg = np.column_stack((sin2outer, sin2outer, zeros))
signal_df = pd.DataFrame(sig_eeg)
signal_df.to_csv("Results-Generation/EyesClosedNovel_Subject3.csv",
                 index=True, header=False, sep="\t")
# Outer
plt.plot(t_out, eyes_closed_outer)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Real EEG (DNF OUTER)")
plt.figure()
plt.plot(t_out, sin2outer)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Time Domain Fake EEG (DNF OUTER)")
plt.legend(["Real EEG", "Sine EEG"])
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(sin2outer, fs, nperseg=fs)
plt.plot(freqs, psd_1, lw=2)
plt.title("Fake EEG FR (DNF OUTER)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(eyes_closed_outer, fs, nperseg=fs)
plt.plot(freqs, psd_1, lw=2)
plt.title("Real EEG FR (DNF OUTER)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.show()
