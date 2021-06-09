import numpy as np
import matplotlib.pyplot as plt

# ******************** Main ********************
sampling_rate = 44100
T = 1/sampling_rate
N = sampling_rate*4
t = np.arange(N)*T
MAX_AMP = 5000  # max amp
A = MAX_AMP*np.exp(-t)
pi = np.pi
A = A * np.sin(2*pi*t/2000)
f = 8
phase = 0
x = 2*pi*f*t + phase
signal = A*np.sin(x)
sin_mean = np.mean(signal)
sin_std = np.std(signal)
noise_amp = 0.2
gauss_noise = noise_amp*np.random.normal(0, 1, len(signal))
noise = 1.2 * np.sin(2 * pi * 5.36 * t + 1.2) + 1.4 * np.sin(2 *
                                                             pi * 7.34 * t + 0.3) + 0.9 * np.sin(2 * pi * 9.24 * t - 0.45)
# noise = noise*gauss_noise
alpha = signal * noise + gauss_noise
plt.plot(t, alpha)
plt.title("Fake EEG")
plt.ylabel("Amplitude (a.u.)")
plt.xlabel("Time (s)")
plt.show()
