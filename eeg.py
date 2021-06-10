class EEG():
    def __init__(self, A, sinusoidial, pi, f, t, phi):
        self.signal = A*sinusoidial(2*pi*f*t + phi)
