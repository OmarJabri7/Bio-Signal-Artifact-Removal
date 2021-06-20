# -*- coding: utf-8 -*-
"""
(C) 2018 Wanting Huang <172258368@qq.com>
(C) 2018 Bernd Porr <bernd.porr@glasgow.ac.uk>

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Demo program which plots one experiment (Temple Run) from
subject 20. The plot shows the 3 channels recorded: Fp1,
EMG at the chin and the switch which acts as an electronic
clapper board.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as math
from eegemg import EegEmg



# plots one experiment from one subject into a single plot window by
# adding a 0.5mV offset to the 2 channels to separate them.
def plotData(subj, experiment, t, fp1, chin, trigger):
    yoffset = 0.001
    plt.title("Subj: "+str(subj)+", Exp:"+experiment)
    plt.xlabel("time(s)")
    plt.ylabel("Amplitude in V")
    plt.plot(t, trigger/5000)
    plt.plot(t, fp1+yoffset/2)
    plt.plot(t, chin-yoffset/2)

        

# main program
# plots subject 20 which is also the example subject in the demo video clip
# playing temple run. Clearly the EOG artefacts can be seen while the subject is doing
# rapid eye movements. The jaw/facial muscle activity is pretty tense as well.
# The 3rd channel is the switch held into the video frame as an electronic clapper board.
# In order to separate the traces and avoiding "subplot" a shift has been introduced.
# If you want to plot more experiments and/or subjects just create subplots in a loop.
#
#
pathToData = "../experiment_data"
#
# subject
subject_number = 20
#
# experiment Temple Run (video game)
experiment = "templerun"
#
# creating class which loads the experiment
eegemg = EegEmg(pathToData, subject_number, experiment)
print("All experiments available? ",eegemg.checkIfSubjectHasAllExperiments(pathToData, subject_number))
#
# filtering out interference
eegemg.filterData()
#
# plot it: you see three traces: the "EEG" from Fp1, the EMG from the chin and the switch
# acting as a "clapper board" to be able to sync video & data.
plotData(subject_number, experiment, eegemg.t, eegemg.fp1, eegemg.chin, eegemg.trigger)
#
# show the experiment
plt.show()
