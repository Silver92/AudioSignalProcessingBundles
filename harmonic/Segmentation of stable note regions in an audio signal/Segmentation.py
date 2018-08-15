import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
import utilFunctions as UF
import harmonicModel as HM
import stft

eps = np.finfo(float).eps

"""
Segmentation of stable note regions in an audio signal

I compute the function identify the stable regions of notes in a specific 
monophonic audio signal. The function returns an array of segments where each segment contains the 
starting and the ending frame index of a stable note.

The input argument to the function are the wav file name including the path (inputFile), threshold to 
be used for deciding stable notes (stdThsld), minimum allowed duration of a stable note (minNoteDur), 
number of samples to be considered for computing standard deviation (winStable), analysis window (window), 
window size (M), FFT size (N), hop size (H), error threshold used in the f0 detection (f0et), magnitude 
threshold for spectral peak picking (t), minimum allowed f0 (minf0) and maximum allowed f0 (maxf0). 
The function returns a numpy array of shape (k,2), where k is the total number of detected segments. 
The two columns in each row contains the starting and the ending frame indices of a stable note segment. 

"""

def segmentStableNotesRegions(inputFile = 'sax-phrase-short.wav', stdThsld=10, minNoteDur=0.1, 
                              winStable = 3, window='hamming', M=1024, N=2048, H=256, f0et=5.0, t=-100, 
                              minf0=310, maxf0=650):
    """
    Function to segment the stable note regions in an audio signal
    Input:
        inputFile (string): wav file including the path
        stdThsld (float): threshold for detecting stable regions in the f0 contour
        minNoteDur (float): minimum allowed segment length (note duration)  
        winStable (integer): number of samples used for computing standard deviation
        window (string): analysis window
        M (integer): window size used for computing f0 contour
        N (integer): FFT size used for computing f0 contour
        H (integer): Hop size used for computing f0 contour
        f0et (float): error threshold used for the f0 computation
        t (float): magnitude threshold in dB used in spectral peak picking
        minf0 (float): minimum fundamental frequency in Hz
        maxf0 (float): maximum fundamental frequency in Hz
    Output:
        segments (np.ndarray): Numpy array containing starting and ending frame indices of every 
                               segment.
    """
    fs, x = UF.wavread(inputFile)                               #reading inputFile
    w  = get_window(window, M)                                  #obtaining analysis window    
    f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)  #estimating F0

    # 1. convert f0 values from Hz to Cents
    for i in range(0, len(f0)):
        if f0[i] == 0:
            f0[i] = eps
    f0cent = 1200*np.log2(f0/55.0)
    # 2. create an array containing standard deviation of last winStable samples
    f0dev = np.zeros(len(f0cent)-winStable+2)
    for i in range(0, len(f0cent)-winStable+2):
        f0dev[i] = np.std(f0cent[i: i+winStable])
    # 3. apply threshold on standard deviation values to find indices of the stable points in melody
    
    segindex = np.zeros(len(f0dev))
    for i in range(0, len(f0dev)-1):
        if f0dev[i] <= stdThsld:
            segindex[i+winStable-1] = i+winStable-1
    
    # 4. create segments of continuous stable points such that concequtive stable points belong to
    #    same segment
    segment = np.array_split(segindex,np.where(np.diff(segindex)!=1)[0]+1)
    # 5. apply segment filtering
    segments = np.array([])
    for i in range(0, len(segment)):      
        #print(len(segment[i]),i)
        if len(segment[i]) >= fs*minNoteDur/float(H):
            a = np.array([segment[i][0],segment[i][len(segment[i])-1]])
            segments = np.append(segments,a)
            print(segments)
    segments = np.reshape(segments,(-1,2))
    # plotSpectogramF0Segments(x, fs, w, N, H, f0, segments)
    plotSpectogramF0Segments(x, fs, w, N, H, f0, segments)
    # return()
    return(segments)
## plot the f0 contour and the estimated segments on the spectrogram
def plotSpectogramF0Segments(x, fs, w, N, H, f0, segments):
    """
    Code for plotting the f0 contour on top of the spectrogram
    """
    # frequency range to plot
    maxplotfreq = 1000.0    
    fontSize = 16

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mX, pX = stft.stftAnal(x, fs, w, N, H)                      #using same params as used for analysis
    mX = np.transpose(mX[:,:int(N*(maxplotfreq/fs))+1])
    
    timeStamps = np.arange(mX.shape[1])*H/float(fs)                             
    binFreqs = np.arange(mX.shape[0])*fs/float(N)
    
    plt.pcolormesh(timeStamps, binFreqs, mX)
    plt.plot(timeStamps, f0, color = 'k', linewidth=5)

    for ii in range(segments.shape[0]):
        plt.plot(timeStamps[segments[ii,0]:segments[ii,1]], f0[segments[ii,0]:segments[ii,1]], color = '#A9E2F3', linewidth=1.5)        
    
    plt.autoscale(tight=True)
    plt.ylabel('Frequency (Hz)', fontsize = fontSize)
    plt.xlabel('Time (s)', fontsize = fontSize)
    plt.legend(('f0','segments'))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    ax.set_aspect((xLim[1]-xLim[0])/(2.0*(yLim[1]-yLim[0])))    
    plt.autoscale(tight=True) 
    plt.show()
    
