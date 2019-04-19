import numpy as np
from numpy.fft import fft, fftfreq

import matplotlib.pyplot as plt

def genSamples(dcOffset, frequency, Vpp, signal='sine', adj=4096, Vmax=1.8, samples=2500, cpuFreq=20000):
    totaltime = samples/cpuFreq
    t = np.linspace(0, totaltime, 2500)
    dt = totaltime/samples
    if signal is 'sine':
        return (Vpp * np.sin(2 * np.pi * frequency * t) + dcOffset) * adj / Vmax, dt
    if signal is 'cos':
        return (Vpp * np.cos(2 * np.pi * frequency * t) + dcOffset) * adj / Vmax, dt
    else:
        return np.zeros(2500), cpuFreq

def runSweep(startfreq, stopfreq, samples=50, dcOffset=1, Vpp=0.5):
    sweep = np.linspace(startfreq, stopfreq, samples)
    for freq in sweep:
        # print('Actual frequency: {}'.format(int(freq)))
        CHANNEL_1_SAMPLES, dt = genSamples(dcOffset, freq, Vpp)
        # print('Samples: ', CHANNEL_1_SAMPLES)
        CHANNEL_1_FFT = fft(CHANNEL_1_SAMPLES)
        CHANNEL_1_FREQS = fftfreq(len(CHANNEL_1_FFT), d=dt)
        idx_1 = np.argmax(np.abs(CHANNEL_1_FFT[1:len(CHANNEL_1_FFT)//2]))
        freq_1 = CHANNEL_1_FREQS[idx_1]
        # print('idx: ', idx_1, 'freq: ', freq_1)
        # freqAccuracy(freq_1, int(freq))
        print('[{:05.1f}, {:05.1f}, {:05.2f}%]'.format(freq, freq_1, freqAccuracy(freq_1, freq)))

def freqAccuracy(guess, actual):
    accuracy = (1-(np.abs(guess-actual)/actual))*100
    # print('{:05.2f}% accuracy.\n'.format(accuracy))
    return accuracy


runSweep(500, 45000, samples=1000)

# CHANNEL_1_SAMPLES, dt = genSamples(1, 500, 0.5)
# print('Samples: ', CHANNEL_1_SAMPLES)
#
# CHANNEL_1_FFT = fft(CHANNEL_1_SAMPLES)
#
# CHANNEL_1_FREQS = fftfreq(len(CHANNEL_1_FFT), d=dt)
#
# idx_1 = np.argmax(np.abs(CHANNEL_1_FFT[1:]))
#
# freq_1 = CHANNEL_1_FREQS[idx_1]
#
# print('idx: ', idx_1, 'freq: ', freq_1)
#
# # Plot the positive spectrum of the FFT
#
# # plt.plot(2.0/self.nsamples * np.abs(sigfft[1:self.nsamples//2]))
#
# length = len(CHANNEL_1_FFT)//2
# plt.plot(CHANNEL_1_FREQS[1:length], 2.0/length * np.abs(CHANNEL_1_FFT[1:length]))
# plt.show()


# CHANNEL_0 = np.zeros(NUM_SAMPLES)
# CHANNEL_1 = np.zeros(NUM_SAMPLES)
# CHANNEL_2 = np.zeros(NUM_SAMPLES)
# CHANNEL_3 = np.zeros(NUM_SAMPLES)
#
# CHANNEL_0_FFT = np.zeros(NUM_SAMPLES)
# CHANNEL_1_FFT = np.zeros(NUM_SAMPLES)
# CHANNEL_2_FFT = np.zeros(NUM_SAMPLES)
# CHANNEL_3_FFT = np.zeros(NUM_SAMPLES)

# io = pruio_new(PRUIO_DEF_ACTIVE, 0, 0, 0)
# IO = io.contents
# if  IO.Errr: raise AssertionError("pruio_new failed (%s)" % IO.Errr)
#
# if pruio_config(io, 50000, 0b11110, 20001, 0):
#     raise AssertionError("config failed (%s)" % IO.Errr)
#
# while True:
#     if pruio_mm_start(io, 0, 0, 0, 0):
#         raise AssertionError("mm_start failed (%s)" % IO.Errr)
#
#     AdcV = IO.Adc.contents.Value
#
#     for i in range(NUM_SAMPLES):
#         CHANNEL_0[i] = AdcV[(i * 4)]
#         CHANNEL_1[i] = AdcV[(i * 4) + 1]
#         CHANNEL_2[i] = AdcV[(i * 4) + 2]
#         CHANNEL_3[i] = AdcV[(i * 4) + 3]
#
#
#     CHANNEL_0_FFT = fft(CHANNEL_0)/NUM_SAMPLES
#     CHANNEL_1_FFT = fft(CHANNEL_1)/NUM_SAMPLES
#     CHANNEL_2_FFT = fft(CHANNEL_2)/NUM_SAMPLES
#     CHANNEL_3_FFT = fft(CHANNEL_3)/NUM_SAMPLES
#
#     CHANNEL_0_FREQS = fftfreq(len(CHANNEL_0_FFT))
#     CHANNEL_1_FREQS = fftfreq(len(CHANNEL_1_FFT))
#     CHANNEL_2_FREQS = fftfreq(len(CHANNEL_2_FFT))
#     CHANNEL_3_FREQS = fftfreq(len(CHANNEL_3_FFT))
#
#     idx_0 = np.argmax(np.abs(CHANNEL_0_FFT))
#     idx_1 = np.argmax(np.abs(CHANNEL_1_FFT))
#     idx_2 = np.argmax(np.abs(CHANNEL_2_FFT))
#     idx_3 = np.argmax(np.abs(CHANNEL_3_FFT))
#
#     freq_0 = CHANNEL_0_FREQS[idx_0]
#     freq_1 = CHANNEL_1_FREQS[idx_1]
#     freq_2 = CHANNEL_2_FREQS[idx_2]
#     freq_3 = CHANNEL_3_FREQS[idx_3]
#
#     freq_0 = abs(freq_0 * NUM_SAMPLES)
#     freq_1 = abs(freq_1 * NUM_SAMPLES)
#     freq_2 = abs(freq_2 * NUM_SAMPLES)
#     freq_3 = abs(freq_3 * NUM_SAMPLES)
#
#     print(np.mean(CHANNEL_0), np.mean(CHANNEL_1), np.mean(CHANNEL_2), np.mean(CHANNEL_3))
#
# pruio_destroy(io)