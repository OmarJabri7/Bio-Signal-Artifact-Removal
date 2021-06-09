#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun  5 04:35:39 2021

@author: omar
"""
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU


class Discriminator():
    def __init__(self):
        super().__init__()

    def model(self):
        model = Sequential()
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def optimizer(self):
        opt = Adam(lr=0.0002, beta_1=0.5)
        return opt


class Generator():
    def __init__(self):
        super().__init__()

    def model(self):
        model = Sequential()
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='sigmoid'))
        return model

    def optimizer(self):
        opt = Adam(lr=0.0002, beta_1=0.5)
        return opt
