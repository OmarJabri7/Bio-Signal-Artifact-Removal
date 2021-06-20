class EEG():
    def __init__(self, A, sinusoidial, pi, f, t, phi):
        self.A = A
        self.f = f
        self.t = t
        self.phi = phi
        self.signal = A*sinusoidial(2*pi*f*t + phi)
