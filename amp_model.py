import numpy as np

def amp_model(prm, x):

    c1 = 14.974 + 1j * 0.0519
    c3 = -23.0954 + 1j * 4.968
    c5 = 21.3936 + 1j * 0.4305

    y = x * (c1 + c3*np.abs(x)**2 + c5*np.abs(x)**4)
    return y
