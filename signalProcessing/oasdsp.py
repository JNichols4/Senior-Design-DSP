import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, correlate
import bisect


def runxcorr(nChSignals, TOA_samples, nmics, nsamples, timestart, endtime, debug=0):
    # Run a cross correlation, in time, across all available channels.
    # Returns the cross correlation values of the combinations.
    matlist = []
    matlist1 = []
    for i in range(nmics):
        templist = []
        templist1 = []
        for j in range(nmics):
            # The meanint variable may need to be a function of the number of samples taken. Current setting is for
            # 1000 samples.
            meanint = 10

            # Re-adjust the input signal based upon the rolling mean. Calculate an averaged signal for both channels.
            xconvolve1 = np.convolve(nChSignals[i], np.ones((meanint,)) / meanint, mode='valid')
            xconvolve2 = np.convolve(nChSignals[j], np.ones((meanint,)) / meanint, mode='valid')

            if debug:
                plt.plot(xconvolve1)
                plt.plot(xconvolve2)
                plt.show()

            # Calculate the correlation between the two averaged signals.
            xcorrelate = correlate(xconvolve1, xconvolve2, mode='full')

            # Take the hilbert transform of the correlate data.
            xcorrenvelope = np.abs(hilbert(xcorrelate))

            # Find the index of the maximum correlation value in the correlation envelope.
            xcorrmaxpos = np.argmax(xcorrenvelope)

            # Generate a set of time to relate the correlation indexes to.
            tcorrelate = np.linspace(0, len(xcorrelate) - 1, num=len(xcorrelate))

            # Find the center value of the time correlate data.
            tcorr_center = len(tcorrelate) / 2

            # Subtract 1 from every tcorrelate value and multiply by the center value.
            tcorrelate = tcorrelate - np.ones(len(tcorrelate)) * tcorr_center

            # Run the bisection of the two time datasets
            p = bisect.bisect(tcorrelate, TOA_samples.item((i, j)))
            templist1.append(p)

            if debug:
                plt.plot(xcorrenvelope)
                plt.show()
            templist.append(xcorrmaxpos)

        matlist.append(templist)
        matlist1.append(templist1)

    xcorrmaxmat = np.matrix(matlist)
    tcorrmat = np.matrix(matlist1)

    subs = np.subtract(xcorrmaxmat, tcorrmat)
    minrow = subs.min(0)
    subs = np.subtract(subs.min(0), minrow.min())
    subs = np.divide(subs, (nsamples / (endtime - timestart)))

    return subs

def computeDirection(rMinTimeRow, nmics, micpos, c=330):
    dpos = []
    miclist = []
    for i in range(nmics):
        miclist.append(i)

    minpos = np.argmin(rMinTimeRow)
    # print('\nminpos\n', minpos)

    # This is based off of the quadrants, make the mics match these quadrants: mic1, q1, mic2, q4, mic3, q2, mic4, q3
    if minpos > 2:
        const = 180
    else:
        const = 0

    # Compute the difference in distances between the first detect microphone and the other mics.
    ipos = micpos[minpos]
    miclist.remove(minpos)
    for i in miclist:
        dpos.append(np.subtract(micpos[i], ipos))

    dposmat = np.matrix([dpos[0], dpos[1], dpos[2]])
    # print('\ndposmat\n', dposmat)

    # Take the pseudo-inverse of the difference in distance matrix.
    pseudoinv = np.linalg.pinv(dposmat)
    # print('\npseudoinv\n', pseudoinv)

    # Compute the c*dT matrix for the calculation.
    cdTmat = np.delete(rMinTimeRow, minpos)
    cdTmat = np.multiply(cdTmat, c)
    cdTmat = np.transpose(cdTmat)
    # print(cdTmat)

    # Calculate the dot product of the two matrices. Gives us the [u;v] matrix.
    uv = np.dot(pseudoinv, cdTmat)
    # print('\nuv\n', uv)

    # Calculate the angle of the guess, adjust the values for the tangent function.
    theta = np.rad2deg(np.arctan2(uv[1], uv[0]))
    if theta < 0:
        theta = theta + 180
    elif theta >=0:
        theta = theta - 180

    # Get theta minimum. Since it is the only item in the ndarray, we can use min() to extract the value from the array
    # and cast it as other types.
    print('theta: {}'.format(theta.min()))

def computeTOASamplesMatrix(micpos, nmics, timestart, endtime, nsamples,c=330):
    arrayTOA = []
    for i in range(nmics):
        temparray = []
        for j in range(nmics):
            temparray.append(distance(micpos[i], micpos[j]) /
                                (c * (1 / (nsamples / (endtime - timestart)))))
        arrayTOA.append(temparray)
    return np.matrix(arrayTOA)

def distance(pos1, pos2):
    # pos1 and pos2 need to be provided as [x1, y1] and [x2, y2]
    dist = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
    # print(dist)
    return dist


"""
A. The general process for this program should be the following:
  Before starting the process, make sure to record the following:
  1) Number of microphones and microphone positions. Everything should be in meters. Device center is the origin.
        nmics -> int, micpos -> [[mic1x, mic1y],[mic2x, mic2y],[mic3x, mic3y],[mic4x, mic4y]]
        
  2) If available, the angle of the noise relative to the device for computation comparison purposes.
        actual = np.rad2deg(np.arctan2(sourceposy, sourceposx)), generates on a scale from [-pi, pi]
        
  3) TOA matrix, of the expected time of arrival between all mics. A function of distance and speed of sound.
        Use the function computeTOASamplesMatrix. Make sure to provide micpos and endtime, timestart. Timestart can be
        set to 0, but endtime needs to be a function of the sampling rate and delay time between samples.
        Ex. 1.5ksps at a delay time of 166667ns results in an endtime of 0.25 seconds. Make sure to inclue the number
            of samples in computeTOASamplesMatrix.

  After setting up the preliminary information:
  1) Generate waveforms with a time delay, or sample for waveforms with a time delay.
        nChSignals -> [[waveform1],[waveform2],[waveform3],[waveform4]]
  2) Run the cross correlation of the data.
        subs -> returned matrix of the minimum row, the equivalent delta samples values. Used for direction calc.
  3) Run the direction computation.
        theta -> guess of the relative angle of the signal. 
"""

# actual = np.rad2deg(np.arctan2(sourcepos[1], sourcepos[0]))
# print('{},{},{},{}'.format(sourcepos[0], sourcepos[1], theta.min(), actual))