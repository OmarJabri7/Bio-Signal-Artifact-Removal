# -*- coding: utf-8 -*-
"""
(C) 2018 Wanting Huang <172258368@qq.com>
(C) 2018 Bernd Porr <bernd.porr@glasgow.ac.uk>

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

API for the data which loads, filters and exports
the EEG/EMG data.

It's also able to return a chunk of EEG data which
is artefact free and chunks of EEG which are polluted with
artefacts.

I can check if all experiments are available.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as math
import scipy.stats as stats


# class which loads the dataset and filters out known interference
class EegEmg:

    EXPERIMENTS = ["lie_relax","blink","eyescrunching","raisingeyebrows","jaw","readingsitting","readinglieing","flow","sudoku","wordsearch","templerun"]
    fs=1000
    
    # specify the path to the data, the subject number and the experiment
    def __init__(self,root_dir,_subj,_experiment):
        self.subj = _subj
        self.experiment = _experiment
        self.subjdir = root_dir+"/"+("subj%02d" % _subj)+"/"
        self.expdir = self.subjdir+self.experiment+"/"

        self.data=np.loadtxt(self.expdir+"emgeeg.dat")
        self.zero_data=np.loadtxt(self.expdir+"zero_time_data.dat")
        self.zero_video=np.loadtxt(self.expdir+"zero_time_video.dat")
        self.artefact=np.loadtxt(self.expdir+"artefact.dat")
        self.relaxed=np.loadtxt(self.expdir+"dataok.dat")

        self.t=self.data[:,0]          #timestamp or sample # (sampling rate fs=1kHz)
        self.fp1=self.data[:,1]        #fp1
        self.chin=self.data[:,2]        #chin
        self.trigger=self.data[:,3]    #switch 

        #AMPLIFER GAIN IS 500, SAMPLING RATE IS 1kHz
        self.fp1=self.fp1/500
        self.chin=self.chin/500
        self.T=1/self.fs
        self.t=np.arange(0,self.T*len(self.fp1),self.T)

    # checks if a subject has all experiments specified in the array self.experiments, i.e. that you can safely run a loop
    # through all experiments
    def checkIfSubjectHasAllExperiments(self,root_dir,subj):
        allok = root_dir+"/"+("subj%02d" % subj)+"/all_exp_ok.dat"
        criterion=np.loadtxt(allok, dtype=bytes).astype(str)
        return criterion == 'True'

    # filters the data from the two channels. Highpass at 0.5Hz, notches at 50Hz, 80Hz and 25Hz
    def filterData(self):
        # smooth it. This gives effectively a higher resolution for the EEG
        bLP,aLP = signal.butter(4,100/self.fs*2)
        self.fp1 = signal.lfilter(bLP,aLP,self.fp1);

        ## highpass at 0.5Hz and 50Hz notch
        bfilt50hz,afilt50hz = signal.butter(4,[48/self.fs*2,52/self.fs*2],'stop')
        bhp,ahp = signal.butter(4,0.5/self.fs*2,'high')
        self.fp1 = signal.lfilter(bhp,ahp,signal.lfilter(bfilt50hz,afilt50hz,self.fp1));
        self.chin = signal.lfilter(bhp,ahp,signal.lfilter(bfilt50hz,afilt50hz,self.chin));

        ## strange 80Hz interference
        bfilt80hz,afilt80hz = signal.butter(4,[78/self.fs*2,82/self.fs*2],'stop')
        self.fp1 = signal.lfilter(bfilt80hz,afilt80hz,self.fp1);
        self.chin = signal.lfilter(bfilt80hz,afilt80hz,self.chin);

        ## strange 25 Hz interference
        bfilt25hz,afilt25hz = signal.butter(4,[24/self.fs*2,26/self.fs*2],'stop')
        self.fp1 = signal.lfilter(bfilt25hz,afilt25hz,self.fp1);
        self.chin = signal.lfilter(bfilt25hz,afilt25hz,self.chin);

    # returns a section where there is no artefact
    def get_artefact_free_EEG_section(self):
        dt=self.zero_data-self.zero_video
        t1=int(self.fs*(self.relaxed[0]+dt))
        t2=int(self.fs*(self.relaxed[1]+dt))
        eegOK=self.fp1[t1:t2]
        return eegOK

    # returns an array of eeg artfact sections
    def get_EEG_artefacts(self):
        tbeginVideo=self.artefact[:,0]
        tendVideo=self.artefact[:,1]
        dt=self.zero_data-self.zero_video
        tbegin=tbeginVideo+dt
        tend=tendVideo+dt
        artefacts=[]
        for i in range(len(tbegin)):
            t1=tbegin[i]
            t2=tend[i]
            t1=int(self.fs*t1)
            t2=int(self.fs*t2)
            yArtefact=self.fp1[t1:t2]
            artefacts.append(yArtefact)
        return artefacts
