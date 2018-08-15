import numpy as np

def DFT(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function returns a numpy array of length N
        X (numpy array) = The N point DFT of the input sequence x
    """
    N = x.size
    X = np.array([])
    for k in range(N):
      s = np.exp(1j * 2 * np.pi * k/N *np.arange(N))
      X = np.append(X, sum(x*np.conjugate(s)))
      absX = abs(X)                                           # compute ansolute value of positive side
      absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
      mX = 20 * np.log10(absX) 
      pX = np.unwrap(np.angle(X)) 
    return(mX, pX)
