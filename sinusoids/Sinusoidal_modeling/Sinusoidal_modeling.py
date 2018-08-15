import numpy as np
from scipy.signal import get_window
import sys, os
import stft
import utilFunctions as UF
import sineModel as SM
import matplotlib.pyplot as plt

"""
Sinusoidal modeling of a multicomponent signal

Perform a sinusoidal analysis of a complex synthetic signal. 
"""
def exploreSineModel(inputFile='multiSines.wav'):
    """
    Input:
            inputFile (string) = wav file including the path
    Output: 
            return True
    """
    window='hamming'                            # Window type
    M=2001                                      # Window size in sample
    N=2048                                      # FFT Size
    t=-80                                       # Threshold                
    minSineDur=0.02                             # minimum duration of a sinusoid
    maxnSines=150                               # Maximum number of sinusoids at any time frame
    freqDevOffset=10                            # minimum frequency deviation at 0Hz
    freqDevSlope=0.001                             # slope increase of minimum frequency deviation
    Ns = 512                                    # size of fft used in synthesis
    H = 128                                     # hop size (has to be 1/4 of Ns)
    
    fs, x = UF.wavread(inputFile)               # read input sound
    w = get_window(window, M)                   # compute analysis window

    # analyze the sound with the sinusoidal model
    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

    # synthesize the output sound from the sinusoidal representation
    y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

    # output sound file name
    outputFile = os.path.basename(inputFile)[:-4] + '_sineModel.wav'

    # write the synthesized sound obtained from the sinusoidal synthesis
    UF.wavwrite(y, fs, outputFile)

    # create figure to show plots
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(3,1,1)
    plt.plot(np.arange(x.size)/float(fs), x)
    plt.axis([0, x.size/float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')
                
    # plot the sinusoidal frequencies
    plt.subplot(3,1,2)
    if (tfreq.shape[1] > 0):
        numFrames = tfreq.shape[0]
        frmTime = H*np.arange(numFrames)/float(fs)
        tfreq[tfreq<=0] = np.nan
        plt.plot(frmTime, tfreq)
        plt.axis([0, x.size/float(fs), 0, maxplotfreq])
        plt.title('frequencies of sinusoidal tracks')

    # plot the output sound
    plt.subplot(3,1,3)
    plt.plot(np.arange(y.size)/float(fs), y)
    plt.axis([0, y.size/float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()
    plt.show()
    return True
