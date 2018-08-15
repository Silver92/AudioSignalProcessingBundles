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
Compute amount of inharmonicity present in a sound
"""
def estimateInharmonicity(inputFile = 'piano.wav', t1=0.1, t2=0.5, window='hamming', 
                            M=2048, N=2048, H=128, f0et=5.0, t=-90, minf0=130, maxf0=180, nH = 10):
    """
    Function to estimate the extent of inharmonicity present in a sound
    Input:
        inputFile (string): wav file including the path
        t1 (float): start time of the segment considered for computing inharmonicity
        t2 (float): end time of the segment considered for computing inharmonicity
        window (string): analysis window
        M (integer): window size used for computing f0 contour
        N (integer): FFT size used for computing f0 contour
        H (integer): Hop size used for computing f0 contour
        f0et (float): error threshold used for the f0 computation
        t (float): magnitude threshold in dB used in spectral peak picking
        minf0 (float): minimum fundamental frequency in Hz
        maxf0 (float): maximum fundamental frequency in Hz
        nH (integer): number of integers considered for computing inharmonicity
    Output:
        meanInharm (float or np.float): mean inharmonicity over all the frames between the time interval 
                                        t1 and t2. 
    """    
    
    # Read the audio file
    (fs,x) = UF.wavread(inputFile)
    w = get_window(window, M)
    harmDevSlope=0.01
    minSinDur=0.0
    # Use harmonic model to to compute the harmonic frequencies and magnitudes
    hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSinDur)
    # Extract the segment in which you need to compute the inharmonicity. 
    l1 = int(np.ceil(t1*fs/H))
    l2 = int(np.ceil(t2*fs/H))
    # Compute the mean inharmonicity for the segment
    Imean = 0
    d = np.array([])
    a = np.array([])
    frame = np.array([],ndmin=2)
    for i in range(l1, l2):
        R = nH
        I = 0
        for r in range(0, R):
            I += abs((hfreq[i][r]-(r+1)*hfreq[i][0]))/(r+1)
        I = I/R
        Imean += I
    Imean = Imean/(l2-l1)
    
    return(Imean)
