import numpy as np

Te = 1
T = 100
sigma_Q = 1
sigma_px = 30
sigma_py = 30

Q = sigma_Q*np.array([[Te**3/ 3, Te**2/2, 0, 0],
              [Te**2/2, Te, 0, 0],
              [0, 0, Te**3/3, Te**2/2],
              [0, 0, Te**2/2, Te]], dtype='float64'
             )

R = np.array([[sigma_px**2, 0],
              [0, sigma_py**2]], dtype='float64'
             )

F = np.array([[1, Te, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, Te],
              [0, 0, 0, 1]], dtype='float64'
             )

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]], dtype='float64')


x_init = np.array([3, 40, -4, 20], dtype='float64')
x_kalm = x_init

P_kaml = np.identity(4)