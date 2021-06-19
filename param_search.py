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
plt.style.use('ggplot')
# ******************** Main ********************


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


def band_limited_noise(noise, noise_lvl, min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = noise
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(noise_lvl*f)


fs = 512
T = 1/fs
N = 20000
t = np.arange(N)*T
MAX_AMP = 1e3  # max amp
A = MAX_AMP*np.exp(-t)
f = 8
A = A * np.sin(2*np.pi*f/fs)
pi = np.pi
f_start = 8
f_end = 10
f_tot = np.linspace(f_start, f_end, N)
f_final = np.cumsum(f_tot/fs)
f = 8
phase = 0
x = 2*pi*f_final*t + phase
signal = A*np.sin(x)
sin_mean = np.mean(signal)
sin_std = np.std(signal)
noise_amp = 0.5
gauss_noise = noise_amp*np.random.normal(0, 1, len(signal))
noise = 1.2 * np.sin(2 * pi * 5.36 * t + 1.2) + 1.4 * np.sin(2 *
                                                             pi * 7.34 * t + 0.3) + 0.9 * np.sin(2 * pi * 9.24 * t - 0.45)
# noise = noise*gauss_noise
alpha = signal * noise + gauss_noise
for s in range(len(alpha)):
    if(s % 1000 == 0):
        alpha[s] *= 1e4
eeg_dat = np.fromfile('subj27/blink.dat')
alpha = alpha + eeg_dat[:len(alpha)]
alpha = sig.detrend(alpha)
mat = loadmat('Data/subject_04.mat')
mat_data = mat["SIGNAL"]
signal = mat_data[:, 1:17]
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
RMS_n = sqrt(RMS_s**2/pow(10, 20/10))
STD_n = RMS_n
AWGN = np.random.normal(0, STD_n, sin2eeg.shape[0])
sin2eeg += AWGN


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

    def sinfunc(t, A, w, p, c): return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    def fitfunc(t): return A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}


dic = fit_sin(t, signal_eeg)
sin2eeg = dic["amp"]*np.sin(dic["omega"]*t + dic["phase"]) + dic["offset"]
eeg_dat = np.fromfile('subj27/blink.dat')
noise_freq = 50  # Hz
eeg_noise = eeg_dat[:len(sin2eeg)]
eeg_noise_fft = np.fft.fft(sin2eeg)
eeg_noise_amp = 2*np.abs(eeg_noise_fft)/N
yn = eeg_noise_amp*np.sin(2*pi*noise_freq*t)
f_sin = 8
sines = 1.2 * np.sin(2 * pi * f_sin * t + 1.2) + 1.4 * np.sin(2 *
                                                              pi * f_sin * t + 0.3)
noise_amp = 0.01
gauss_noise = noise_amp*np.random.normal(0, 1, len(sin2eeg))
noise_lvl = 1
freqs = autocorr(eeg_dat)
eeg_noise = band_limited_noise(eeg_noise, noise_lvl, 40, 50, N, fs)
sin2eeg = sin2eeg + eeg_noise  # + AWGN
sin_peaks = sig.find_peaks(sin2eeg)[0]
# sin2eeg[sin_peaks] = 2
plt.plot(signal_eeg)
plt.plot(sin2eeg)
plt.figure()
win = 4 * fs
freqs, psd_1 = sig.welch(sin2eeg, fs, nperseg=fs)
plt.plot(freqs, psd_1, color='k', lw=2)
plt.title("Fake EEG FR")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.figure()
fft_sin = np.fft.fft(sin2eeg)  # FFT of signal
# alpha_amp = 2*np.abs(fft_alpha)/N  # Amplitude of signal
sin_amp = np.abs(np.fft.fft(fft_sin/N))**2
sin_hz = np.linspace(0, fs/2, floor(N/2) + 1)
plt.plot(sin_hz, sin_amp[:len(sin_hz)])
plt.plot(eeg_freqs, amp_eeg[:len(eeg_freqs)])
plt.xlabel("Frequencies (Hz)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Frequency Spectrum of Real VS Fake")
plt.show()
# %%
# # Band Pass Filter between 8-12 Hz
# filt_alpha = eeg
# nfft = len(filt_alpha)
# fft_alpha = np.fft.fft(filt_alpha)  # FFT of signal
# # alpha_amp = 2*np.abs(fft_alpha)/N  # Amplitude of signal
# alpha_amp = np.abs(np.fft.fft(sig.detrend(filt_alpha)/N))**2
# f_spec, t_spec, sxx = sig.spectrogram(filt_alpha, fs)
# plt.pcolormesh(t_spec, f_spec, sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.figure()
# # Frequencies on axis in Hz of signal
# alpha_hz = np.linspace(0, fs/2, floor(N/2) + 1)
# alpha_phi = 2*np.angle(fft_alpha)/N  # Phase of EEG Alpha wave
# # %%
# plt.plot(t, filt_alpha)
# plt.title("Real EEG (FP1)")
# plt.ylabel("Voltage (uV)")
# plt.xlabel("Time (s)")
# plt.figure()
# plt.plot(alpha_hz, alpha_amp[:len(alpha_hz)])
# plt.xlabel("Frequencies (Hz)")
# plt.ylabel("Amplitude (a.u.)")
# plt.title("Frequency Spectrum of Alpha EEG wave")
# plt.figure()
# win = 4 * fs
# freqs, psd_2 = sig.welch(filt_alpha, fs, nperseg=win)
# plt.plot(freqs, psd_2, color='k', lw=2)
# plt.title("Real EEG FR")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectral density (V^2 / Hz)')
# plt.figure()
# plt.plot(t, alpha)
# plt.title("FAKE EEG")
# plt.figure()
# win = 4 * fs
# freqs, psd_1 = sig.welch(alpha, fs, nperseg=win)
# plt.plot(freqs, psd_1, color='k', lw=2)
# plt.title("Fake EEG FR")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectral density (V^2 / Hz)')
# plt.figure()
# nfft = len(alpha)
# fft_alpha = np.fft.fft(alpha)
# alpha_amp = 2*np.abs(fft_alpha)/N
# alpha_hz = np.linspace(0, int(fs/2), floor(int(N/2)) + 1)
# # %%
# plt.plot(alpha_hz, alpha_amp[:len(alpha_hz)])
# plt.xlabel("Frequencies (Hz)")
# plt.ylabel("Amplitude (a.u.)")
# plt.title("Frequency Spectrum of Alpha EEG wave")
# plt.show()
