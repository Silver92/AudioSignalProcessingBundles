import numpy as np

def IDFT(X):
    """
    Input:
        X (numpy array) = frequency spectrum (length N)
    Output:
        The function returns a numpy array of length N 
        x (numpy array) = The N point IDFT of the frequency spectrum X
    """

    N = X.size
    x = np.array([])
    for k in range(N):
      s = np.exp(1j * 2 * np.pi * k/N * np.arange(N))
      x = np.append(x, 1.0/N * sum(X*s))
    return(x)
