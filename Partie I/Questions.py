import matplotlib.pyplot as plt
import numpy as np

from variables import *
from main import *


def question4():
    X = creer_trajectoire(F, Q, x_init, T)
    Y = creer_observation(H, R, X)

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle('Trajectoire et observation', fontsize=18)
    axs[0].plot(X[0], X[2], label='Vraie trajectoire', color='orange', alpha=0.7)
    axs[0].set_title('X')
    axs[0].plot(Y[0], Y[1],  label='Trajectoire observée', color='blue', alpha=0.7)

    for ax in axs.flat:
        ax.legend()

    plt.show()


def question7():
    n = 200
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)
    all_P = np.zeros((P_kaml.shape[0], P_kaml.shape[1], n), dtype='float64')
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)

    for i in range(1, n):
        [result_matrix[i], all_P[:, :, i]] = filtre_de_kalman(
            F, Q, H, R,
            np.transpose(all_obs)[i],
            result_matrix[i-1],
            all_P[:, :, i-1]
        )

    final_result_matrix = np.transpose(result_matrix)

    fig = plt.figure()
    fig.suptitle('Erreur moyenne : ' + str(average_err(final_result_matrix, all_evol)))
    plt.plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
    plt.plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
    plt.legend()
    plt.show()


def question9():
    n = 200
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)

    all_P = np.zeros((P_kaml.shape[0], P_kaml.shape[1], n), dtype='float64')
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)

    for i in range(1, n):
        [result_matrix[i], all_P[:, :, i]] = filtre_de_kalman(
            F, Q, H, R,
            np.transpose(all_obs)[i],
            result_matrix[i-1],
            all_P[:, :, i-1]
        )

    final_result_matrix = np.transpose(result_matrix)
    fig, axs = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle('Erreur moyenne sur la position ' + str(average_err(final_result_matrix, all_evol)), fontsize=18)

    axs[0].plot(all_evol[0], 'r', label='x_real', alpha=0.7)
    axs[0].plot(final_result_matrix[0], label='x_est', alpha=0.7)
    axs[0].set_title('Suivi de trajectoire sur x', fontsize=16)

    axs[1].plot(all_evol[2], 'r', label='y_real', alpha=0.7)
    axs[1].plot(final_result_matrix[2], label='y_est', alpha=0.7)
    axs[1].set_title('Suivi de trajectoire sur y', fontsize=16)

    axs[2].plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
    axs[2].plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
    axs[2].set_title('Suivi de la position', fontsize=16)

    for ax in axs.flat:
        ax.legend()

    fig.tight_layout()
    fig.show()


#question4()
#question7()
question9()







