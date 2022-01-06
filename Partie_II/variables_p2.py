import numpy as np
import scipy.io


Te = 1
sigma_r = 50
sigma_o = 50
sigma_Q = 800

vecteur_x = scipy.io.loadmat('fichiers_donnees/vecteur_x_avion_ligne.mat')['vecteur_x']
vecteur_y = scipy.io.loadmat('fichiers_donnees/vecteur_y_avion_ligne.mat')['vecteur_y']
x_init = np.array([3, 40, -4, 20], dtype='float64')

F = np.array([[1, Te, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, Te],
              [0, 0, 0, 1]], dtype='float64'
             )

Q = np.array([
    [Te**3/ 3, Te**2/2, 0, 0],
    [Te**2/2, Te, 0, 0],
    [0, 0, Te**3/3, Te**2/2],
    [0, 0, Te**2/2, Te]], dtype='float64'
)*sigma_Q

R = np.array([[(np.pi*sigma_o/180)**2, 0], [0, sigma_r**2]], dtype='float64')

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]], dtype='float64')

P_kaml = np.identity(4)



