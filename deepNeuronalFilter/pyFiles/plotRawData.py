#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:33:19 2020

@author: Sama
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig
import fir1

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 5})

plt.close("all")
fs = 250

subject = 1

EEG_DAT = np.loadtxt(
    '../SubjectData/EEG_Subject{}.tsv'.format(subject))
data_length = len(EEG_DAT)
innerElectSignal = EEG_DAT[0:data_length, 1]
outerElectNoise = EEG_DAT[0:data_length, 2]

print(max(innerElectSignal))
print(max(outerElectNoise))

freqXRaw_CLOSED = np.linspace(0, fs/2, int(data_length/2))

fft_innerElectSignal = np.abs(np.fft.fft(
    innerElectSignal)[0:np.int(data_length/2)])
fft_outerElectNoise = np.abs(np.fft.fft(
    outerElectNoise)[0:np.int(data_length/2)])

dc = int((0.5/fs)*data_length)
low50 = int((48/fs)*data_length)
high50 = int((52/fs)*data_length)

fft_innerElectSignal[0:dc] = 0
fft_outerElectNoise[0:dc] = 0
fft_innerElectSignal[low50:high50] = 0
fft_outerElectNoise[low50:high50] = 0

myFig0 = plt.figure('raw fig')

ax5 = myFig0.add_subplot(425)
plt.plot(outerElectNoise, linewidth=0.4)
plt.title('raw outer eyes closed')
ax6 = myFig0.add_subplot(426)
plt.plot(freqXRaw_CLOSED, fft_outerElectNoise, linewidth=0.4)

ax7 = myFig0.add_subplot(427)
plt.plot(innerElectSignal, linewidth=0.4)
plt.title('raw inner eyes closed')
ax8 = myFig0.add_subplot(428)
plt.plot(freqXRaw_CLOSED, fft_innerElectSignal, linewidth=0.4)

fs = 250
cf = (np.arange(0, 126, 1)/fs)*2
filterDelay = 999
DC_removal = (sig.firwin(
    filterDelay, [cf[2]], window='hanning', pass_zero=False))*1000
DC_removal_outer_open = fir1.Fir1(DC_removal)
DC_removal_inner_open = fir1.Fir1(DC_removal)
DC_removal_outer_closed = fir1.Fir1(DC_removal)
DC_removal_inner_closed = fir1.Fir1(DC_removal)

flat_outer_open = np.zeros(data_length)
flat_inner_open = np.zeros(data_length)
flat_outer_closed = np.zeros(data_length)
flat_inner_closed = np.zeros(data_length)

for i in range(data_length):
    flat_outer_closed[i] = DC_removal_outer_closed.filter(
        outerElectNoise[i])
    flat_inner_closed[i] = DC_removal_inner_closed.filter(
        innerElectSignal[i])

flat_outer_closed = flat_outer_closed[filterDelay::]
flat_inner_closed = flat_inner_closed[filterDelay::]

dataLengthShort = len(flat_outer_open)

freqXDC_CLOSED = np.linspace(0, fs/2, int(dataLengthShort/2))
freqCutDC = 1
fft_flat_outer_closed = np.abs(np.fft.fft(flat_outer_closed)[
                               0:np.int(dataLengthShort/2)])
fft_flat_inner_closed = np.abs(np.fft.fft(flat_inner_closed)[
                               0:np.int(dataLengthShort/2)])

dc_flat = int((0/fs)*dataLengthShort)
low50_flat = int((48/fs)*dataLengthShort)
high50_flat = int((52/fs)*dataLengthShort)

fft_flat_inner_closed[0:dc_flat] = 0
fft_flat_outer_closed[0:dc_flat] = 0
fft_flat_inner_closed[low50_flat:high50_flat] = 0
fft_flat_outer_closed[low50_flat:high50_flat] = 0

ax5 = myFig0.add_subplot(425)
plt.plot(flat_outer_closed, linewidth=0.4)
plt.title('outer')
ax6 = myFig0.add_subplot(426)
plt.plot(freqXDC_CLOSED[freqCutDC::],
         fft_flat_outer_closed[freqCutDC::], linewidth=0.4)

ax7 = myFig0.add_subplot(427)
plt.plot(flat_inner_closed, linewidth=0.4)
plt.title('inner')
ax8 = myFig0.add_subplot(428)
plt.plot(freqXDC_CLOSED[freqCutDC::],
         fft_flat_inner_closed[freqCutDC::], linewidth=0.4)
