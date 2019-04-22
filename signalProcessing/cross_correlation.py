import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def correlateSignals(signals, micpos, opfile):
    # Takes an array of signals and computes the cross-correlation of each combination.
    # The signals array is a m x n array where m is the number of microphones

    nmics = len(signals)

    corrDict = {}

    for inc in range(nmics):
        for cap in range(nmics):
            templist = []
            corr = np.correlate(signals[inc], signals[cap], 'full')
            templist.append(corr)
            templist.append(np.argmax(corr))
            corrDict[str(inc)+str(cap)] = templist
    temparray = []
    lc=0
    A = []
    for heading in corrDict:
        temparray.append(corrDict[heading][1]) # value of correlation
        lc+=1
        if lc > 3:
            A.append(temparray)
            temparray=[]
            lc=0
    corrMatrix = np.matrix(A)
    # print(corrMatrix)
    computeDirection(corrMatrix, micpos, opfile)

def magnitude(array):
    temp = []
    for e in array:
        temp.append(np.sqrt(np.real(e)^2+np.imag(e)^2))
    return temp

def computeDirection(corrMatrix, micpos, opfile, c=330, samples=1000):

    dpos = []

    adjMat = np.subtract(corrMatrix, np.min(corrMatrix))
    adjMat = np.divide(adjMat, samples)
    # print('\nadjMat\n', adjMat)

    minRow = adjMat.min(0)
    # print('\nminRow\n', minRow)

    rTimeMinRow = np.multiply(minRow, 2)
    opfile.write('rTimeMinRow: \n{}\n'.format(rTimeMinRow))
    # print('\nrTimeMinRow\n', rTimeMinRow)

    minpos = np.argmin(minRow)
    # print('\nminpos\n', minpos)

    ipos = micpos[minpos]
    miclist = [0,1,2,3]
    miclist.remove(minpos)
    for i in miclist:
        dpos.append(np.subtract(micpos[i], ipos))

    dposmat = np.matrix([dpos[0], dpos[1], dpos[2]])
    # opfile.write('dposmat: {}\n'.format(dposmat))
    # print('\ndposmat\n', dposmat)

    pseudoinv = np.linalg.pinv(dposmat)
    # print('\npseudoinv\n', pseudoinv)

    cdTmat = np.delete(rTimeMinRow, minpos)
    cdTmat = np.multiply(cdTmat, c)
    cdTmat = np.transpose(cdTmat)
    # print(cdTmat)

    uv = np.dot(pseudoinv, cdTmat)
    # print('uv\n', uv)

    const = 0

    # if uv[0]<0: # x
    #     #     if uv[1]<0: # y
    #     #         const = 180
    #     # if uv[0]<0:
    #     #     if uv[1]>0:
    #     #         const = 0
    #     # if uv[0]>0:
    #     #     if uv[1]>0:
    #     #         const = 0
    #     # if uv[0]>0:
    #     #     if uv[1]<0:
    #     #         const = 180

    theta = np.rad2deg(np.arctan(uv[1]/uv[0])) + const
    opfile.write(str(theta.min())+'\n')


