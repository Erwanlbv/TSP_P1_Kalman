import matplotlib.pyplot as plt
import numpy as np

from Partie_I.main_p1 import *


from main_p2 import *
from variables_p2 import *


def question7():
    vecteur_x_2 = creer_trajectoire(F, Q, x_init, 100) #Format 4*100
    obs_cart_traj = creer_observation(H, R, vecteur_x_2) # Observation au format linéraire, donc pas forcément le plus représentatif
    obs_pol_traj = cart_to_pol(obs_cart_traj[0], obs_cart_traj[1])

    pred_x_matrix = np.zeros((len(obs_pol_traj[0]), 4)) #Format 100*4
    all_P = np.zeros((100, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')

    pred_x_matrix[0] = np.transpose(vecteur_x_2)[0]

    for i in range(1, len(obs_pol_traj[0])):
        [pred_x_matrix[i], all_P[i]] = filtre_kalman_radar(
            F, Q, R, np.transpose(obs_pol_traj)[i], pred_x_matrix[i-1], all_P[i-1]
        )

    final_result_matrix = np.transpose(pred_x_matrix) #Retour au format 4*100 pour l'affichage

    fig, axs = plt.subplots(3, 1, figsize=(13, 7))
    fig.suptitle('Erreur vectorielle moyenne : ' + 'str(avg_erro_vect(final_result_matrix, all_evol))')
    axs[0].plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='Trajectoire estimée', alpha=0.7)
    axs[0].plot(vecteur_x_2[0], vecteur_x_2[2], color='orange', label='Trajectoire réelle', alpha=0.7)
    axs[0].scatter(obs_cart_traj[0], obs_cart_traj[1], color='orange', label='Trajectoire mesurée', alpha=0.7)

    axs[1].plot(final_result_matrix[0], color='blue', label='x_est', alpha=0.7)
    axs[1].plot(vecteur_x_2[0], color='orange', label='x_reel', alpha=0.7)

    axs[2].plot(final_result_matrix[2], color='blue', label='y_est', alpha=0.7)
    axs[2].plot(vecteur_x_2[2], color='orange', label='y_reel', alpha=0.7)

    for ax in axs.flat:
        ax.legend()

    fig.show()


question7()