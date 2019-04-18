import numpy as np
import scipy.signal as sps
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import random

class Channels:
    def __init__(self, nmics, frequencies, phases, nsamples=1000, timestart=0, endtime=1, a=0.4, g=0.3):

        self.nmics = nmics
        self.frequencies = frequencies
        self.phases = phases

        self.nsamples = nsamples
        self.timestart = timestart
        self.endtime = endtime
        self.a = a
        self.g = g

        self.nChSignals = [[0 for x in range(nsamples)] for y in range(nmics)]
        self.nChNoise = [[0 for x in range(nsamples)] for y in range(nmics)]
        self.nChPSD = [[0 for x in range(nsamples)] for y in range(nmics)]
        self.nChNoisePSD = [[0 for x in range(nsamples)] for y in range(nmics)]
        self.avgChPSD = [0 for x in range(nsamples)]
        self.avgChNoisePSD = [0 for x in range(nsamples)]
        self.wMask = [0 for x in range(nsamples)]

        self.dt = (self.endtime-self.timestart)/self.nsamples

        self.classSetup()

    def classSetup(self):
        self.setChannelSignals()
        self.setChannelNoise()
        self.calcChannelPSDs()
        self.calcChannelNoisePSD()
        self.avgChannelPSD()
        self.avgChannelNoisePSD()

        self.wMask = self.weightedMask(self.avgChPSD, self.avgChNoisePSD, self.a, self.g)


    def sine(self, frequency, phase, samples=1000, timestart=0, endtime=1, noise=False):
        t = np.linspace(timestart, endtime, samples)
        signal = np.sin(2*np.pi*frequency*t+phase)
        if noise:
            signoise = 0.0008*np.asarray(random.sample(range(0,1000),samples))
            return signal+signoise
        else:
            return signal

    def cosine(self, frequency, phase, samples=1000, timestart=0, endtime=1, noise=False):
        t = np.linspace(timestart, endtime, samples)
        signal = np.cos(2*np.pi*frequency*t+phase)
        if noise:
            signoise = 0.0008*np.asarray(random.sample(range(0,1000),samples))
            return signal+signoise
        else:
            return signal

    def PSD(self, fftsignal, timestep, totaltime):
        # Computes the spectral power of each component in fftsignal.
        return ((np.abs(fftsignal) * timestep) ** 2) / totaltime

    def weightedMask(self, psdSignal, psdNoise, a, g):
        # Computes the masking weight between two channels.
        l = len(psdSignal)
        wMask = []
        for i in range(l):
            if psdSignal[i] <= psdNoise[i]:
                wMask.append(max(0.1, (psdSignal[i]-a*psdNoise[i])/psdSignal[i]))
            else:
                wMask.append(max(0.1, (psdSignal[i]-a*psdNoise[i])/psdSignal[i])*(psdSignal[i]/psdNoise[i])**g)
        return wMask

    def setChannelSignals(self):
        phases = self.phases
        for i in range(self.nmics):
            signal = [0 for x in range(self.nsamples)]
            phase = phases.pop(0)
            for freq in self.frequencies:
                signal = signal + self.sine(freq, phase, samples=self.nsamples, timestart=self.timestart,
                                            endtime=self.endtime, noise=True)
            self.nChSignals[i] = signal

    def calcChannelPSDs(self):
        for i in range(self.nmics):
            self.nChPSD[i] = self.PSD(fft(self.nChSignals[i]), self.dt, self.endtime-self.timestart)

    def calcChannelNoisePSD(self):
        for i in range(self.nmics):
            self.nChNoisePSD[i] = self.PSD(fft(self.nChNoise[i]), self.dt, self.endtime-self.timestart)

    def randomNoise(self, noiseLevel=3):
        return 3*0.0008*np.asarray(random.sample(range(0,1000),self.nsamples))

    def avgChannelPSD(self):
        for i in range(self.nsamples):
            csum = 0
            for j in range(self.nmics):
                csum = csum + self.nChPSD[j][i]
            self.avgChPSD[i] = csum/self.nmics

    def setChannelNoise(self):
        for i in range(self.nmics):
            self.nChNoise[i] = self.randomNoise()

    def avgChannelNoisePSD(self):
        for i in range(self.nsamples):
            csum = 0
            for j in range(self.nmics):
                csum = csum + self.nChNoisePSD[j][i]
            self.avgChNoisePSD[i] = csum/self.nmics

    def plotChannels(self, show=True):
        for i in range(self.nmics):
            plt.plot(self.nChSignals[i])
        if show:
            plt.show()

    def plotWeightedMask(self, show=True):
        plt.plot(self.wMask[:int(self.nsamples/2)])
        if show:
            plt.show()

    def plotChannelFFT(self, channel=0, show=True):
        sigfft = fft(self.nChSignals[channel])
        plt.plot(2.0/self.nsamples * np.abs(sigfft[1:self.nsamples//2]))
        if show:
            plt.show()

    def mag(self, item):
        return np.sqrt(np.real(item)**2+np.imag(item)**2)

    def wCrossCorrelationFunction(self, channelx, channely):
        cf = lambda k: (self.wMask[k]**2 * self.nChPSD[channelx][k] * np.conj(self.nChPSD[channely][k]))/\
                       (self.mag(self.nChPSD[channelx][k])*self.mag(self.nChPSD[channely][k])) * \
                        np.exp(2*np.pi*k/self.nsamples)
        wCorr = []
        for i in range(0, self.nsamples-1):
            wCorr.append(cf(i))

        return lambda t: sum(wCorr)*np.exp(t)



OAS = Channels(4, [5, 10, 15, 20], [0, np.pi/4, np.pi/6, np.pi/8])

OAS.plotChannelFFT(channel=0, show=False)
OAS.plotWeightedMask()

opFunc = OAS.wCrossCorrelationFunction(channelx=0, channely=1)
myvars = []
for i in range(OAS.nsamples):
    myvars.append(opFunc(i*OAS.dt))
plt.plot(myvars)

opFunc = OAS.wCrossCorrelationFunction(channelx=1, channely=3)
myvars = []
for i in range(OAS.nsamples):
    myvars.append(opFunc(i*OAS.dt))
plt.plot(myvars)

plt.show()