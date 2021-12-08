import numpy as np

from variables_p2 import *


def cart_to_pol(x, y): #Format 1*2
    return np.array([np.sqrt(x**2 + y**2), np.arctan(y/x)])

#Puisque nous sommes dans un cas où x n'est jamais négatif nous pouvons utiliser cette formule pour les
#coordonnées polaires, dans un cas réel il faudrait utiliser 2*np.arctan(y/(x + np.sqrt(x**2, y**2)))


def cree_observations_radar(R, vecteur_x): #Prend un vecteur_x de dimension 4*n
    return cart_to_pol(vecteur_x[0, :], vecteur_x[2, :]) + np.transpose(np.random.multivariate_normal(np.zeros(2), R, len(vecteur_x[0])))
#R n'est pas défini comme dans l'énonce, ici R[0,0] correspond à l'icertitude sur la mesure de r, R[1, 1] sur celle de theta

#Question 4 : Voir feuille
#Question 5 : Voir feuille pour justification choix de xk|k-1 et expression de la nouvelle formule

def filtre_kalman_radar(F, Q, R, y_k, x_kalm_prec, P_kalm_prec):

    #Partie Prédiction :
    x_pred_suiv = F.dot(x_kalm_prec)
    P_pred_suiv = F.dot(P_kalm_prec.dot(np.transpose(F))) + Q

    #Partie filtrage :
    H = np.array([
        [x_pred_suiv[0]/(np.sqrt(x_pred_suiv[0]**2 + x_pred_suiv[2]**2)), 0, x_pred_suiv[2]/np.sqrt(x_pred_suiv[0]**2 + x_pred_suiv[2]**2), 0],
        [-x_pred_suiv[2]/(x_pred_suiv[0]**2 + x_pred_suiv[2]**2), 0, x_pred_suiv[1]/(x_pred_suiv[0]**2 + x_pred_suiv[2]**2), 0]
    ])

    inv_matrix = np.linalg.inv(H.dot(P_pred_suiv.dot(np.transpose(H))) + R)
    K_filt_suiv = P_pred_suiv.dot(np.transpose(H).dot(inv_matrix))
    P_kalm_suiv = (np.eye(4, dtype='float64') - K_filt_suiv.dot(H)).dot(P_pred_suiv)

    y_k = y_k + H.dot(x_pred_suiv) - np.transpose(cart_to_pol(x_pred_suiv[0], x_pred_suiv[2]))
    x_kalm_suiv = x_pred_suiv + K_filt_suiv.dot(y_k - H.dot(x_pred_suiv))

    #On pourrait directement faire le -H.dot() dans l'expression de y_k, mais la transition est plus visible ainsi.

    return [x_kalm_suiv, P_kalm_suiv]







