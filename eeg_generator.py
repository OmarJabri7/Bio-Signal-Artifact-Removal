import numpy as np
import os
from numpy import fft
import scipy.signal as sig
from scipy.signal import butter, lfilter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack


NOISE_SRC = ["blink", "eyescrunching", "flow",
             "readingsitting", "sudoku",
             "templerun", "wordsearch", "jaw", "turninghead", "lie_relax",
             "raisingeyebrows", "readinglieing"]


class EEG_GEN():

    def __init__(self) -> None:
        self.alphas = {}
        self.deltas = {}
        self.noises = {}
        self.fs = 1000.0
        self.pre_amp = 500
        self.corrfactor = 2

    def folders_in(self, path_to_parent):
        for fname in os.path.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent, fname)):
                yield os.path.join(path_to_parent, fname)

    def load_alpha_data(self, dir, data_subj, eeg_subj, noise_source):
        idx = 1
        if("+" in noise_source):
            noises = noise_source.split("+")
            artefact = 0
            noise_lengths = []
            for noise in noises:
                self.load_alpha_data(dir, data_subj, eeg_subj, noise)
            for noise in noises:
                noise_lengths.append(
                    self.get_noise_length(str(eeg_subj), noise))
            N = self.get_min_samples_alpha(noise_lengths)
            for noise in noises:
                if(self.folders_in(dir)):
                    self.artefact = np.loadtxt(
                        f"{dir}/subj{data_subj}/{noise}/emgeeg.dat")
                    artefact += self.artefact[:N, idx]
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
                        f"{dir}/subj{data_subj}/{noise_source}.dat")
                    self.artefact = self.artefact
        else:
            if(self.folders_in(dir)):
                self.artefact = np.loadtxt(
                    f"{dir}/subj{data_subj}/{noise_source}/emgeeg.dat")
                if(eeg_subj in self.noises):
                    if(noise_source in self.noises[eeg_subj]):
                        self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                        self.samples_alpha = self.artefact[:, idx].shape[0]
                    else:
                        self.noises[eeg_subj][noise_source] = {}
                        self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                        self.samples_alpha = self.artefact[:, idx].shape[0]
                else:
                    self.noises[eeg_subj] = {}
                    self.noises[eeg_subj][noise_source] = {}
                    self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                    self.samples_alpha = self.artefact[:, idx].shape[0]

            else:
                self.artefact = np.loadtxt(
                    f"{dir}/subj{data_subj}/{noise_source}.dat")
                self.artefact = self.artefact

    def load_delta_data(self, dir, data_subj, eeg_subj, noise_source):
        idx = 1
        if("+" in noise_source):
            noises = noise_source.split("+")
            artefact = 0
            noise_lengths = []
            for noise in noises:
                self.load_delta_data(dir, data_subj, eeg_subj, noise)
            for noise in noises:
                noise_lengths.append(
                    self.get_noise_length(str(eeg_subj), noise))
            N = self.get_min_samples_delta(noise_lengths)
            for noise in noises:
                if(self.folders_in(dir)):
                    self.artefact = np.loadtxt(
                        f"{dir}/subj{data_subj}/{noise}/emgeeg.dat")
                    artefact += self.artefact[:N, idx]
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
                        f"{dir}/subj{data_subj}/{noise_source}.dat")
                    self.artefact = self.artefact
        else:
            if(self.folders_in(dir)):
                self.artefact = np.loadtxt(
                    f"{dir}/subj{data_subj}/{noise_source}/emgeeg.dat")
                if(eeg_subj in self.noises):
                    if(noise_source in self.noises[eeg_subj]):
                        self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                        self.samples_delta = self.artefact[:, idx].shape[0]
                    else:
                        self.noises[eeg_subj][noise_source] = {}
                        self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                        self.samples_delta = self.artefact[:, idx].shape[0]
                else:
                    self.noises[eeg_subj] = {}
                    self.noises[eeg_subj][noise_source] = {}
                    self.noises[eeg_subj][noise_source] = self.artefact[:, idx]
                    self.samples_delta = self.artefact[:, idx].shape[0]
            else:
                self.artefact = np.loadtxt(
                    f"{dir}/subj{data_subj}/{noise_source}.dat")
                self.artefact = self.artefact

    def get_sampling_rate(self, dir, data_subj, eeg_subj, noise_source):
        if(self.folders_in(dir)):
            self.artefact = np.loadtxt(
                f"{dir}/subj{data_subj}/{noise_source}/emgeeg.dat")
            ts = self.artefact[:, 0]
            self.fs = 1/(ts[1] - ts[0])
        else:
            self.artefact = np.loadtxt(
                f"{dir}/subj{data_subj}/{noise_source}.dat")
            ts = self.artefact[:, 0]
            self.fs = 1/(ts[1] - ts[0])

    def get_min_samples_alpha(self, noise_samples):
        self.samples_alpha = min(noise_samples)
        return self.samples_alpha

    def get_min_samples_delta(self, noise_samples):
        self.samples_delta = min(noise_samples)
        return self.samples_delta

    def butter_bandpass(self, signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def butter_bandstop(self, signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandstop')
        y = lfilter(b, a, signal)
        return y

    def butter_lowpass(self, signal, cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def butter_highpass(self, signal, cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def gen_sine_alpha(self, A, f, freqs, fs=1000, bandpass=False, sum_sines=False, optimal=False) -> None:
        x = np.arange(self.samples_alpha)
        self.fs = fs
        self.alpha = A * np.sin(2 * np.pi * f * x / fs)
        if(sum_sines):
            self.alpha += 10 * np.sin(2 * np.pi * freqs[0] * x / fs)
            self.alpha += 12 * np.sin(2 * np.pi * freqs[1] * x / fs)
            self.alpha += 15 * np.sin(2 * np.pi * freqs[3] * x / fs)
            self.alpha += 10 * np.sin(2 * np.pi * freqs[4] * x / fs)
        if(bandpass):
            self.alpha = self.butter_bandpass(
                self.alpha, freqs[0] - 1, freqs[4] + 1, self.fs, 2)
        if(optimal):
            pass

    def gen_sine_delta(self, A, f, freqs, fs=1000, bandpass=False, sum_sines=False, optimal=False) -> None:
        x = np.arange(self.samples_delta)
        self.fs = fs
        self.delta = A * np.sin(2 * np.pi * f * x / fs)
        if(sum_sines):
            self.delta += 10 * np.sin(2 * np.pi * freqs[0] * x / fs)
            self.delta += 12 * np.sin(2 * np.pi * freqs[1] * x / fs)
            self.delta += 15 * np.sin(2 * np.pi * freqs[3] * x / fs)
            self.delta += 10 * np.sin(2 * np.pi * freqs[4] * x / fs)
        if(bandpass):
            self.delta = self.butter_bandpass(
                self.delta, freqs[0] - 1, freqs[4] + 1, self.fs, 2)
        if(optimal):
            pass

    def set_noise(self, eeg_subj, noise_source, noise_new):
        self.noises[eeg_subj][noise_source] = noise_new

    def gen_eeg_alpha(self, eeg_subj, noise_source) -> None:
        if(eeg_subj in self.alphas):
            if(noise_source in self.alphas[eeg_subj]):
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.alphas[eeg_subj][noise_source] = self.alpha + \
                    tmp_noise[:self.samples_alpha]
            else:
                self.alphas[eeg_subj][noise_source] = {}
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.alphas[eeg_subj][noise_source] = self.alpha + \
                    tmp_noise[:self.samples_alpha]
        else:
            self.alphas[eeg_subj] = {}
            self.alphas[eeg_subj][noise_source] = {}
            tmp_noise = self.noises[eeg_subj][noise_source]
            self.alphas[eeg_subj][noise_source] = self.alpha + \
                tmp_noise[:self.samples_alpha]

    def gen_eeg_delta(self, eeg_subj, noise_source) -> None:
        if(eeg_subj in self.deltas):
            if(noise_source in self.deltas[eeg_subj]):
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.deltas[eeg_subj][noise_source] = self.delta + \
                    tmp_noise[:self.samples_delta]
            else:
                self.deltas[eeg_subj][noise_source] = {}
                tmp_noise = self.noises[eeg_subj][noise_source]
                self.deltas[eeg_subj][noise_source] = self.delta + \
                    tmp_noise[:self.samples_delta]
        else:
            self.deltas[eeg_subj] = {}
            self.deltas[eeg_subj][noise_source] = {}
            tmp_noise = self.noises[eeg_subj][noise_source]
            self.deltas[eeg_subj][noise_source] = self.delta + \
                tmp_noise[:self.samples_delta]

    def gen_alpha_optimal(self, data_subj, eeg_subj, noise_source):
        self.alpha_data = np.loadtxt(
            f"{dir}/subj{data_subj}/{noise_source}/emgeeg.dat")
        self.alpha_eeg = self.alpha_data[:, 1]
        self.t = self.alpha_data[:, 0]
        self.fit_sin(self.t, self.alpha_eeg)

    def gen_delta_optimal():
        pass

    def fit_sin(self, tt, yy):
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
        opt_res = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. /
                   f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
        self.optimal_alpha = opt_res["amp"] * np.sin(
            2 * np.pi * opt_res["omega"] * self.t + opt_res["phase"]) + opt_res["offset"]

    def gen_gain(self, signal) -> float:
        signal = np.abs(signal)
        str_signal = str(signal)
        str_signal = str_signal.split(".")
        digits = str_signal[0]
        n = len(digits)
        return 1/(10**(n))

    def save_data(self, eeg_data, noise_data, subj, sig_type):
        sig_fake = np.column_stack((eeg_data, noise_data))
        signal_df = pd.DataFrame(sig_fake)
        signal_df.to_csv(
            f"deepNeuronalFilter/SubjectData/{sig_type}/EEG_Subject{subj}.tsv", index=True, header=False, sep="\t")
        print(f"{sig_type} Signals Saved!")

    def save_xls(self, list_dfs, xls_path, sheet_names):
        with pd.ExcelWriter(xls_path) as writer:
            i = 0
            for df in (list_dfs):
                df.to_excel(writer, f"{sheet_names[i]}", index=False)
                i += 1
            writer.save()

    def get_eeg_alpha(self, eeg_subj, noise_source) -> np.array:
        return self.alphas[eeg_subj][noise_source]

    def get_eeg_delta(self, eeg_subj, noise_source) -> np.array:
        return self.deltas[eeg_subj][noise_source]

    def get_noise(self, eeg_subj, noise_source) -> np.array:
        if(noise_source == "all"):
            return self.noises[eeg_subj]
        return self.noises[eeg_subj][noise_source]

    def get_pure_alpha(self) -> np.array:
        return self.alpha

    def get_pure_delta(self) -> np.array:
        return self.delta

    def get_noise_length(self, eeg_subj, noise_source):
        return len(self.noises[eeg_subj][noise_source])

    def band_limited_noise(self, noise, noise_lvl, min_freq, max_freq, samples=1024, samplerate=1):
        freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
        f = noise
        idx = np.where(np.logical_and(freqs < min_freq, freqs > max_freq))[0]
        f[idx] = 1

        def fftnoise(f):
            f = np.array(f, dtype='complex')
            Np = (len(f) - 1) // 2
            phases = np.random.rand(Np) * 2 * np.pi
            phases = np.cos(phases) + 1j * np.sin(phases)
            f[1:Np+1] *= phases
            f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
            return np.fft.ifft(f).real
        return fftnoise(noise_lvl*f)

    def calc_SNR(self, clean, noisy, sig_type, snr_fct, calc=0) -> float:
        N = len(clean)
        if(sig_type == 0):
            aS = 8
            aE = 12
            start = int((aS / self.fs) * N)
            end = int((aE / self.fs) * N)
            diff_noise = end-start
        elif(sig_type == 1):
            dS = 1
            dE = 5
            start = int((dS / self.fs) * N)
            end = int((dE / self.fs) * N)
            diff_noise = end-start
        if(snr_fct == 0):
            noisy_pow = 1/N * (np.sum(noisy**2))
            clean_pow = 1/N * (np.sum(clean**2))
        elif(snr_fct == 1):
            clean_fft = np.abs(np.fft.fft(clean)/len(clean))
            clean_fft = (clean_fft[range(int(len(clean)/2))])
            noisy_fft = np.abs(np.fft.fft(noisy)/len(noisy))
            noisy_fft = noisy_fft[range(int(len(noisy)/2))]
            noisy_pow = (np.sum(noisy_fft[start:end]))
            clean_pow = (np.sum(clean_fft[start:end]))
        elif(snr_fct == 2):
            clean_fft = np.abs(np.fft.fft(clean)[
                0: np.int(N / 2)])
            noisy_fft = np.abs(np.fft.fft(noisy)[
                0: np.int(N / 2)])
            noisy_pow = (np.sum(noisy_fft[start:end]**2))
            clean_pow = (np.sum(clean_fft[start:end]**2))
        if(calc == 0):
            return (clean_pow)/np.abs(noisy_pow - clean_pow)
        elif(calc == 1):
            return clean_pow/noisy_pow

    def plot_psd(self, signal):
        freqs, psd = sig.welch(signal)
        plt.semilogx(freqs, psd)
        plt.title('PSD: power spectral density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.figure()

    def plot_spectrogram(self, signal):
        freqs, times, spectrogram = sig.spectrogram(signal)
        plt.plot(freqs, spectrogram)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.figure()

    def plot_freq_resp(self, signal, Fs, title, data_subj):
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
        plt.ylabel('Amplitude (' + r'$\mu$V/Hz)')
        plt.savefig(f"Results-Generation/subj{data_subj}/Frequency_{title}")
        fig = plt.figure()
        return fig

    def plot_freq_resp_vs(self, sig_type, signal, target, Fs, title, signal1, signal2, data_subj):
        if(sig_type == 0):
            start = 8
            end = 12
        elif(sig_type == 1):
            start = 1
            end = 5
        fourierTransform = np.fft.fft(signal)/len(signal)
        fourierTransform = fourierTransform[range(int(len(signal)/2))]
        tpCount = len(signal)
        values = np.arange(int(tpCount/2))
        timePeriod = tpCount/Fs
        frequencies = values/timePeriod
        fourierTransform2 = np.fft.fft(target)/len(target)
        fourierTransform2 = fourierTransform2[range(int(len(target)/2))]
        tpCount2 = len(target)
        values2 = np.arange(int(tpCount2/2))
        timePeriod2 = tpCount2/Fs
        frequencies2 = values2/timePeriod2
        idx = (np.where(np.logical_and(frequencies >= 0, frequencies <= end)))
        idx2 = (np.where(np.logical_and(
            frequencies >= 0, frequencies <= end)))
        plt.plot(frequencies[idx[0][0]:idx[0]
                 [len(idx[0]) - 1]], abs(fourierTransform[idx[0][0]:idx[0]
                                                          [len(idx[0]) - 1]]))
        plt.plot(frequencies2[idx2[0][0]:idx2[0]
                 [len(idx2[0]) - 1]], abs(fourierTransform2[idx2[0][0]:idx2[0]
                                                            [len(idx2[0]) - 1]]))
        plt.title(title)
        plt.xlim(start - 1,end + 1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (' + r'$\mu$V/Hz)')
        plt.legend([f"{signal1}", f"{signal2}"])
        plt.savefig(
            f"Results-Generation/subj{data_subj}/Frequency_Diff_{title}")
        fig = plt.figure()
        return fig

    def plot_welch(self, signal, Fs, title):
        win = Fs
        freqs, psd = sig.welch(signal, Fs)
        plt.plot(freqs, psd, lw=2)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (' + r'$\mu$V/Hz)')
        fig = plt.figure()
        return fig

    def plot_welch_vs(self, signal, target, Fs, title, signal1, signal2):
        win = 4*Fs
        freqs1, psd1 = sig.welch(signal, Fs)
        freqs2, psd2 = sig.welch(target, Fs)
        plt.plot(freqs1, psd1, lw=2)
        plt.plot(freqs2, psd2, lw=2)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.legend([f"{signal1}", f"{signal2}"])
        fig = plt.figure()
        return fig

    def plot_time_series(self, signal, title, data_subj, fig = True, legend = False):
        plt.plot(signal)
        plt.title(title)
        plt.ylabel("Signal Voltage " + r'($\mu$V)')
        plt.xlabel("Time (s)")
        # plt.ylim(-7, 7)
        if(legend == True):
            plt.legend(["DNF Error", "LMS Error", "Laplace Error"])
        if(fig):
            fig = plt.figure()
            plt.savefig(f"Results-Generation/subj{data_subj}/Temporal_{title}")
            return fig

    def plot_all(self, eeg, noise):
        eeg_time = self.plot_time_series(eeg, "Noisy Time Spectrum")
        noise_time = self.plot_time_series(noise, "Noise Time Spectrum")
        eeg_fr = self.plot_freq_resp(eeg, self.fs, "Noisy Frequency Spectrum")
        noise_fr = self.plot_freq_resp(
            noise, self.fs, "Noise Frequency Spectrum")
        plt.show()

    def bar_plot(self, labels, dnf_snr, lms_snr, laplace_snr, signal, data_subj):
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        dnf_max = np.max(dnf_snr)
        dnf_min = np.min(dnf_snr)
        lms_max = np.max(lms_snr)
        lms_min = np.min(lms_snr)
        laplace_max = np.max(laplace_snr)
        laplace_min = np.min(laplace_snr)
        max_snr = max(dnf_max, lms_max, laplace_max)
        min_snr = min(dnf_min, lms_min, laplace_min)
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, dnf_snr,
                        width, label='DNF')
        rects2 = ax.bar(x, lms_snr, width, label='LMS')
        rects3 = ax.bar(x + width/2, laplace_snr, width, label='Laplace')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('SNR Improvements (dB)')
        ax.set_xlabel('Noise Sources')
        ax.set_title(
            f"SNR of DNF, LMS and Laplace Improvements for {signal} waves")
        ax.set_xticks(x)
        plt.ylim(min_snr - 5, max_snr + 5)
        ax.set_xticklabels(labels)
        ax.legend()

        # ax.bar_label(rects1, padding=3)
        # ax.bar_label(rects2, padding=3)
        # ax.bar_label(rects3, padding=3)

        fig.tight_layout()
        plt.savefig(
            f"Results-Generation/subj{data_subj}/Bar_Plot_SNRs_{signal}")
        return fig
