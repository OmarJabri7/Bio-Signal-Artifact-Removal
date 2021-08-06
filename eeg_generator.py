import numpy as np
import os
import scipy.signal as sig
from scipy.signal import butter, lfilter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
import scipy

NOISE_SRC = ["blink", "eyescrunching", "flow",
             "readingsitting", "sudoku",
             "templerun", "wordsearch", "jaw", "turninghead", "lie_relax",
             "raisingeyebrows", "readinglieing"]


class EEG_GEN():

    def __init__(self) -> None:
        self.signals = {}
        self.noises = {}
        self.fs = 1000.0

    def folders_in(self, path_to_parent):
        for fname in os.path.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent, fname)):
                yield os.path.join(path_to_parent, fname)

    def load_data(self, dir, eeg_subj, noise_source):
        if("+" in noise_source):
            noises = noise_source.split("+")
            artefact = 0
            noise_lengths = []
            for noise in noises:
                noise_lengths.append(
                    self.get_noise_length(str(eeg_subj), noise))
            self.get_min_samples(noise_lengths)
            for noise in noises:
                if(self.folders_in(dir)):
                    self.artefact = np.loadtxt(
                        f"{dir}/subj{eeg_subj}/{noise}/emgeeg.dat")
                    artefact += self.artefact[:self.samples, 2]
                    if(eeg_subj in self.noises):
                        if(noise_source in self.noises[eeg_subj]):
                            self.noises[eeg_subj][noise_source] = artefact
                        else:
                            self.noises[eeg_subj][noise_source] = {}
                            self.noises[eeg_subj][noise_source] = artefact
                    else:
                        self.noises[eeg_subj] = {}
                        self.noises[eeg_subj][noise_source] = {}
                        self.noises[eeg_subj][noise_source] = artefact
                else:
                    self.artefact = np.loadtxt(
                        f"{dir}/subj{eeg_subj}/{noise_source}.dat")
                    self.artefact = self.artefact
        else:
            if(self.folders_in(dir)):
                self.artefact = np.loadtxt(
                    f"{dir}/subj{eeg_subj}/{noise_source}/emgeeg.dat")
                if(eeg_subj in self.noises):
                    if(noise_source in self.noises[eeg_subj]):
                        self.noises[eeg_subj][noise_source] = self.artefact[:, 2]
                    else:
                        self.noises[eeg_subj][noise_source] = {}
                        self.noises[eeg_subj][noise_source] = self.artefact[:, 2]
                else:
                    self.noises[eeg_subj] = {}
                    self.noises[eeg_subj][noise_source] = {}
                    self.noises[eeg_subj][noise_source] = self.artefact[:, 2]
            else:
                self.artefact = np.loadtxt(
                    f"{dir}/subj{eeg_subj}/{noise_source}.dat")
                self.artefact = self.artefact

    def get_sampling_rate(self, dir, eeg_subj, noise_source):
        if(self.folders_in(dir)):
            self.artefact = np.loadtxt(
                f"{dir}/subj{eeg_subj}/{noise_source}/emgeeg.dat")
            ts = self.artefact[:, 0]
            self.fs = 1/(ts[1] - ts[0])
        else:
            self.artefact = np.loadtxt(
                f"{dir}/subj{eeg_subj}/{noise_source}.dat")
            ts = self.artefact[:, 0]
            self.fs = 1/(ts[1] - ts[0])

    def get_min_samples(self, noise_samples):
        self.samples = min(noise_samples)

    def butter_bandpass(self, signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def butter_lowpass(self, signal, cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def gen_sin(self, A, f, fs=1000, bandpass=False) -> None:
        x = np.arange(self.samples)
        self.fs = fs
        self.signal = A * np.sin(2 * np.pi * f * x / fs)
        if(bandpass):
            self.signal = self.butter_bandpass(self.signal, 7, 14, self.fs, 6)

    def gen_eeg(self, eeg_subj, noise_source) -> None:
        if(eeg_subj in self.signals):
            if(noise_source in self.signals[eeg_subj]):
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.signals[eeg_subj][noise_source] = self.signal + \
                    tmp_noise[:self.samples]
                # self.signals[eeg_subj][noise_source] = self.butter_bandpass(
                #     self.signal + tmp_noise[:self.samples],  1, 300, self.fs, 2)
            else:
                self.signals[eeg_subj][noise_source] = {}
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.signals[eeg_subj][noise_source] = self.signal + \
                    tmp_noise[:self.samples]
                # self.signals[eeg_subj][noise_source] = self.butter_bandpass(
                #     self.signal + tmp_noise[:self.samples],  1, 300, self.fs, 2)
        else:
            self.signals[eeg_subj] = {}
            self.signals[eeg_subj][noise_source] = {}
            tmp_noise = self.noises[eeg_subj][noise_source]
            self.signals[eeg_subj][noise_source] = self.signal + \
                tmp_noise[:self.samples]
            # self.signals[eeg_subj][noise_source] = self.butter_bandpass(
            #     self.signal + tmp_noise[:self.samples],  1, 300, self.fs, 2)

    def fit_sin(tt, yy):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        # assume uniform spacing
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))
        Fyy = abs(np.fft.fft(yy))
        # excluding the zero frequency "peak", which is related to offset
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array(
            [guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c): return A * np.sin(w*t + p) + c

        popt, pcov = scipy.optimize.curve_fit(
            sinfunc, tt, yy, p0=guess, maxfev=5000)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        def fitfunc(t): return A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}

    def gen_gain(self, signal) -> float:
        signal = np.abs(signal)
        str_signal = str(signal)
        str_signal = str_signal.split(".")
        digits = str_signal[0]
        n = len(digits)
        return 1/(10**(n - 1))

    def save_data(self, eeg_data, noise_data, subj):
        sig_fake = np.column_stack((eeg_data, noise_data[:self.samples]))
        signal_df = pd.DataFrame(sig_fake)
        signal_df.to_csv(
            f"deepNeuronalFilter/SubjectData/EEG_Subject{subj}.tsv", index=True, header=False, sep="\t")
        print("Signals Saved!")

    def get_eeg(self, eeg_subj, noise_source) -> np.array:
        return self.signals[eeg_subj][noise_source]

    def get_noise(self, eeg_subj, noise_source) -> np.array:
        if(noise_source == "all"):
            return self.noises[eeg_subj]
        return self.noises[eeg_subj][noise_source]

    def get_pure(self) -> np.array:
        return self.signal

    def get_noise_length(self, eeg_subj, noise_source):
        return len(self.noises[eeg_subj][noise_source])

    def calc_SNR(self, signal, noise, snr_fct) -> float:
        N = len(signal)
        if(snr_fct == 0):
            noise_pow = 1/N * (np.sum(noise**2))
            sig_pow = 1/N * (np.sum(signal**2))

        elif(snr_fct == 1):
            sig_fft = np.abs(np.fft.fft(signal)[
                0: np.int(N / 2)])
            noise_fft = np.abs(np.fft.fft(noise)[
                0: np.int(N / 2)])
            sig_fft[0] = 0
            noise_pow = (np.sum(noise_fft[1:100]))
            sig_pow = (np.sum(sig_fft[1:100]))
        return (np.abs(sig_pow - noise_pow))/noise_pow

    def plot_psd(self, signal):
        freqs, psd = sig.welch(signal)
        plt.semilogx(freqs, psd)
        plt.title('PSD: power spectral density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        # plt.figure()

    def plot_spectrogram(self, signal):
        freqs, times, spectrogram = sig.spectrogram(signal)
        plt.plot(freqs, spectrogram)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        # plt.figure()

    def plot_freq_resp(self, signal, Fs, title):
        fourierTransform = np.fft.fft(signal)/len(signal)
        fourierTransform = fourierTransform[range(int(len(signal)/2))]
        tpCount = len(signal)
        values = np.arange(int(tpCount/2))
        timePeriod = tpCount/Fs
        frequencies = values/timePeriod
        plt.plot(frequencies, abs(fourierTransform))
        plt.title(title)
        # plt.xlim(0, 200)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (V)')
        plt.savefig(f"Results-Generation/Frequency_{title}")
        # fig = plt.figure()
        # return fig

    def plot_welch(self, signal, Fs, title):
        win = Fs
        freqs, psd = sig.welch(signal, Fs)
        plt.plot(freqs, psd, lw=2)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        # fig = plt.figure()
        # return fig

    def plot_time_series(self, signal, title):
        plt.plot(signal)
        plt.title(title)
        plt.ylabel("Signal Voltage (V)")
        plt.xlabel("Time (s)")
        # plt.ylim(-7, 7)
        plt.savefig(f"Results-Generation/Temporal_{title}")
        fig = plt.figure()
        return fig

    def plot_all(self, eeg, noise):
        eeg_time = self.plot_time_series(eeg, "Noisy Time Spectrum")
        noise_time = self.plot_time_series(noise, "Noise Time Spectrum")
        eeg_fr = self.plot_freq_resp(eeg, self.fs, "Noisy Frequency Spectrum")
        noise_fr = self.plot_freq_resp(
            noise, self.fs, "Noise Frequency Spectrum")
        plt.show()
