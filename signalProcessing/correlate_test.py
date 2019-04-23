import numpy as np
import bisect
from numpy.fft import fft
import random
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert


class OAS:
    def __init__(self, nmics, sourcepos, frequencies, nsamples = 1000, c = 330, timestart=0, endtime=1, datafile=1):
        self.nmics = nmics
        self.micpos = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        self.sourcepos = sourcepos
        self.nsamples = nsamples
        self.frequencies = frequencies
        self.c = c
        self.timestart = timestart
        self.endtime = endtime

        self.datafile = datafile

        self.nChSignals = [[0 for x in range(nsamples)] for y in range(nmics)]

        self.setup()

    def setup(self):
        self.delays = self.generateDelay(self.sourcepos)
        self.TOA_samples = self.computeTOASamplesMatrix()
        self.setChannelSignals()

    def generateDelay(self, sourcepos):
        # sourceposition must be provided. [x, y]
        # find the relative radii from the signal source.
        # returns the delay times from the original source based upon the first contact.
        rR = []
        for pos in self.micpos:
            # radii between source and m1, m2, m3, m4 in that order.
            rR.append(np.sqrt((sourcepos[0] - pos[0]) ** 2 + (sourcepos[1] - pos[1]) ** 2))

        # position of the minimum, lets us know which mic is the one detecting the sound first.
        rRmin = min(rR)
        rAdj = np.subtract(rR, rRmin)
        dTimes = np.divide(rAdj, self.c)

        retList = []
        for item in dTimes:
            retList.append(item)
        # print('delays: {}'.format(retList))
        return retList

    def sine(self, frequency, phase, samples=1000, timestart=0, endtime=1, noise=False):
        t = np.linspace(timestart, endtime, samples)
        signal = np.sin(2*np.pi*frequency*t+phase)
        if noise:
            signoise = 0.0008*np.asarray(random.sample(range(0,1000),samples))
            return signal+signoise
        else:
            return signal

    def setChannelSignals(self):
        delays = self.delays
        phase = 0
        for i in range(self.nmics):
            signal = [0 for x in range(self.nsamples)]
            delay = delays.pop(0)
            for freq in self.frequencies:
                signal = signal + self.sine(freq, phase, samples=self.nsamples, timestart=delay, endtime=self.endtime,
                                            noise=True)
            self.nChSignals[i] = signal

    def plotChannels(self, show=True):
        for i in range(self.nmics):
            plt.plot(self.nChSignals[i])
        if show:
            plt.show()

    def runxcorr(self, debug = 0):
        ofile = open('activesource.txt', 'w')
        # Run a cross correlation, in time, across all available channels.
        # Returns the cross correlation value of the combinations.
        matlist = []
        matlist1 = []
        for i in range(self.nmics):
            templist = []
            templist1 = []
            for j in range(self.nmics):
                xcorrelate = correlate(self.nChSignals[i], self.nChSignals[j], mode='full')
                xcorrelatefft = fft(xcorrelate)
                xcorrenvelope = np.abs(hilbert(xcorrelate))
                xcorrmaxpos = np.argmax(xcorrenvelope)

                tcorrelate = np.linspace(0, len(xcorrelate) - 1, num=len(xcorrelate))
                tcorr_center = len(tcorrelate) / 2

                # fft_max_period = len(y_corr_fft)/np.argmax(abs(y_corr_fft[0:len(y_corr_fft)/2]))
                # corr_index_low[k] = int(corr_index_max  - 1.5*fft_max_period), same for highind just change - to +
                # if corr_index_high[k] > len(ycorr_envelope) -1 :  corr_index_high[k] =  len(ycorr_envelope)- 1

                # fftmaxp = len(xcorrelatefft)/np.argmax(np.abs(xcorrelatefft[0:len(xcorrelatefft)//2]))
                # lowcorrind = int(xcorrmaxpos - 1.5*fftmaxp)
                # if lowcorrind < 0: lowcorrind = 0
                # highcorrind = int(xcorrmaxpos + 1.5*fftmaxp)
                # if highcorrind > len(xcorrenvelope)-1: highcorrind = len(xcorrenvelope)-1

                # xcorr = xcorr - ones(len(xcorr)) * xcorr_center
                tcorrelate = tcorrelate - np.ones(len(tcorrelate)) * tcorr_center


                p = bisect.bisect(tcorrelate, self.TOA_samples.item((i,j)))
                templist1.append(p)

                # if corr_index_low[k] <= p and p <= corr_index_high[
                #     k]:  # only search if it's within range of the central maximum
                #     for i in arange(p, corr_index_high[k], 1):  # search upward from p to len(xo)
                #         if fabs(xcorr[i] - del_TOA_mic_pairs[k, n, m]) <= box_samples:

                if debug:
                    plt.plot(xcorrenvelope)
                    plt.show()
                templist.append(xcorrmaxpos)
            matlist.append(templist)
            matlist1.append(templist1)
        xcorrmaxmat = np.matrix(matlist)
        tcorrmat = np.matrix(matlist1)
        # print(self.TOA_samples)
        # print(xcorrmaxmat)
        # print(tcorrmat)
        subs = np.subtract(xcorrmaxmat, tcorrmat)
        minrow = subs.min(0)
        subs = np.subtract(subs.min(0), minrow.min())
        subs = np.divide(subs, (self.nsamples / (self.endtime - self.timestart)))
        # THIS VALUE OF SUBS IS WHAT WE NEED TO COMPUTE THE DIRECTION!!
        # print('delays: {}\n'.format(subs))
        # print('delays: {}'.format(subs))
        self.computeDirection(subs)
        return xcorrmaxmat

    def computeDirection(self, rMinTimeRow):

        dpos = []

        minpos = np.argmin(rMinTimeRow)
        # print('\nminpos\n', minpos)

        # This is based off of the quadrants, make the mics match the quadrants. mic1, q1, mic2, q4, mic3, q2, mic4, q3
        if minpos > 2:
            const = 180
        else:
            const = 0

        ipos = self.micpos[minpos]
        miclist = [0, 1, 2, 3]
        miclist.remove(minpos)
        for i in miclist:
            dpos.append(np.subtract(self.micpos[i], ipos))

        dposmat = np.matrix([dpos[0], dpos[1], dpos[2]])
        # opfile.write('dposmat: {}\n'.format(dposmat))
        # print('\ndposmat\n', dposmat)

        pseudoinv = np.linalg.pinv(dposmat)
        # print('\npseudoinv\n', pseudoinv)

        cdTmat = np.delete(rMinTimeRow, minpos)
        cdTmat = np.multiply(cdTmat, self.c)
        cdTmat = np.transpose(cdTmat)
        # print(cdTmat)

        uv = np.dot(pseudoinv, cdTmat)
        # print('\nuv\n', uv)

        theta = np.rad2deg(np.arctan2(uv[1], uv[0]))
        if theta < 0:
            theta = theta + 180
        elif theta >=0:
            theta = theta - 180

        actual = np.rad2deg(np.arctan2(self.sourcepos[1], self.sourcepos[0]))
        # print('pos: {}'.format(self.sourcepos))
        # print('theta calc: {}, theta actual: {}\n\n'.format(theta, actual))
        # plt.polar(theta, 1, 'ro')
        # plt.polar(actual, 1, 'ro')
        # plt.show()
        # opfile.write(str(theta.min()) + '\n')

        print('{},{},{},{}'.format(self.sourcepos[0], self.sourcepos[1], theta.min(), actual))

    def computeTOASamplesMatrix(self):
        arrayTOA = []
        for i in range(self.nmics):
            temparray = []
            for j in range(self.nmics):
                temparray.append(self.distance(self.micpos[i], self.micpos[j]) /
                                 (self.c * (1 / (self.nsamples / (self.endtime - self.timestart)))))
            arrayTOA.append(temparray)
        # print(arrayTOA)
        return np.matrix(arrayTOA)

    def distance(self, pos1, pos2):
        # pos1 and pos2 need to be provided as [x1, y1] and [x2, y2]
        dist = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
        # print(dist)
        return dist


def runsignalsweep(mics, xgrid, ygrid, step=2):
    # xgrid, ygrid need to be provided as [xstart, xstop] format
    for x in range(xgrid[0], xgrid[1], step):
        for y in range(ygrid[0], ygrid[1], step):
            newobj = OAS(mics, [x, y], [50, 100, 150, 200])
            newobj.runxcorr()

runsignalsweep(4, [-21, 21], [-21, 21])
#
# OAS = OAS(4, [8, -17], [5, 10, 15, 20])
# OAS.plotChannels()
# OAS.runxcorr()