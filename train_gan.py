import numpy as np
import matplotlib.pyplot as plt
import torch
from gan import Discriminator, Generator
import scipy.io
import tensorflow as tf
import os
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Uncomment for sinusoid:

# N = 1024 #Samples
# signal = torch.zeros((N,2))
# signal[:,0] = 2* np.pi* torch.rand(N)
# signal[:,1] = torch.sin(signal[:,0])


mat = scipy.io.loadmat("Data/subject_00.mat")

data = mat["SIGNAL"]

timestamps = data[:, 0]

signals = data[:, 1:17]

plt.plot(signals[:, np.random.randint(0, 15)])

print(f"Alpha waves shape: {signals.shape}")

# %%

targets = torch.zeros(signals.shape[0], signals.shape[1])

print(f"Target waves shape: {targets.shape}")

train_set = [(signals[:, i], targets[:, i]) for i in range(signals.shape[1])]

# %%
sig_d = Discriminator()
sig_g = Generator()
BUFFER_SIZE = 60000
BATCH_SIZE = 256
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=sig_g.optimizer(),
                                 discriminator_optimizer=sig_d.optimizer(),
                                 generator=sig_g.model(),
                                 discriminator=sig_d.model())
train_dataset = tf.data.Dataset.from_tensor_slices(
    signals).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# %%
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
# %%
# Discriminator
sig_d = Sequential()
sig_d.add(Dense(16))
sig_d.add(LeakyReLU(alpha=0.2))
sig_d.add(Dropout(0.4))
sig_d.add(Dense(32))
sig_d.add(LeakyReLU(alpha=0.2))
sig_d.add(Dropout(0.4))
sig_d.add(Dense(2, activation='sigmoid'))
# Generator
sig_g = Sequential()
sig_g.add(Dense(16))
sig_g.add(LeakyReLU(alpha=0.2))
sig_g.add(Dropout(0.4))
sig_g.add(Dense(32))
sig_g.add(LeakyReLU(alpha=0.2))
sig_g.add(Dropout(0.4))
sig_g.add(Dense(2, activation='sigmoid'))
opt_d = Adam(lr=0.0002, beta_1=0.5)
opt_g = Adam(lr=0.0002, beta_1=0.5)
alpha = signals[:, 0].reshape(-1, 1).astype(int)
sig_d.build(alpha)
sig_d.summary()
# sig_g.summary()
