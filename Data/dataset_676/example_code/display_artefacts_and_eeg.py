# -*- coding: utf-8 -*-
"""
(C) 2018 Wanting Huang <172258368@qq.com>
(C) 2018 Bernd Porr <bernd.porr@glasgow.ac.uk>

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Plots sections of EEG containing artefacts and
one section of EEG which does not contain artefacts.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as math
from eegemg import EegEmg

import scipy.stats as stats


def plotData(subj, experiment, fp1):
    plt.title("Subj: "+str(subj)+", Exp:"+experiment)
    plt.xlabel("time(s)")
    plt.ylabel("Amplitude in V")
    plt.plot(fp1)

        

# Plots the annotated sections of EEG where there was no artefact and
# where there were artefacts.
#
# subject 20
subject_number = 20
#
pathToData = "../experiment_data"
#
f = 1
for experiment in EegEmg.EXPERIMENTS:
    
    # creating class which loads the experiment
    eegemg = EegEmg(pathToData, 20, experiment)
    #
    # filtering out interference
    eegemg.filterData()
    #
    cleanEEG = eegemg.get_artefact_free_EEG_section()
    artefEEG = eegemg.get_EEG_artefacts()
    # plot it: you see three traces: the "EEG" from Fp1, the EMG from the chin and the switch
    # acting as a "clapper board" to be able to sync video & data.
    plt.figure(f)
    plt.subplot(211)
    plotData(subject_number, experiment, cleanEEG)
    plt.subplot(212)
    for i in range(len(artefEEG)):
        plotData(subject_number, experiment, artefEEG[i])
    f = f + 1

# show the experiment
plt.show()
