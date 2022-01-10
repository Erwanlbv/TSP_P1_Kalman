import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from variables_p1 import *
from main_p1 import *


def question4():
    X = creer_trajectoire(F, Q, x_init, T)
    Y = creer_observation(H, R, X)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Trajectoire et observation \n sigma_Q = ' + str(sigma_Q) + '\n' + 'sigma_px = ' + str(sigma_px) +
                                                                      'sigma_py = ' + str(sigma_py),  fontsize=18)
    plt.plot(X[0], X[2], label='Vraie trajectoire', color='orange', alpha=0.7)
    plt.scatter(Y[0], Y[1],  label='Trajectoire observée', color='blue', alpha=0.4)
    plt.legend()
    plt.show()


def question7():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)
    all_P = np.zeros((n, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)
    all_K = np.zeros((n, 4, 2))

    for i in range(1, n):
        [result_matrix[i], all_P[i], all_K[i]] = filtre_de_kalman(
            F, Q, H, R,
            np.transpose(all_obs)[i],
            result_matrix[i-1],
            all_P[i-1]
        )

    final_result_matrix = np.transpose(result_matrix)

    print(" Valeurs pour les prédictions : ")
    print(final_result_matrix.transpose()[:10])
    print('Valeurs des positions exactes')
    print(all_evol.transpose()[:10])
    print("Valeurs pour les K : ")
    print(all_K[:10])

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Erreur vectorielle moyenne : ' + str(avg_erro_vect(final_result_matrix, all_evol)) + '\n' +
                 'sigma_Q = ' + str(sigma_Q) + '\n' + 'sigma_px = ' + str(sigma_px) +' sigma_py =' + str(sigma_py),
                 fontsize=14)

    print("Erreur séparée : ")
    print(avg_error_sep(final_result_matrix, all_evol))
    print('Plus grosse erreur : ')
    print(sup_errr(final_result_matrix, all_evol))

    plt.plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
    plt.plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
    plt.scatter(all_obs[0], all_obs[1], color='red', label='Observations', alpha=0.2)
    plt.legend()
    plt.show()


def question7_pred_only():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)

    for i in range(1, n):
        result_matrix[i] = F.dot(result_matrix[i-1]) + np.transpose(np.random.multivariate_normal(np.zeros(4), Q))

    final_result_matrix = np.transpose(result_matrix)

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Erreur vectorielle moyenne : ' + str(avg_erro_vect(final_result_matrix, all_evol)) + '\n' +
                 'sigma_Q = ' + str(sigma_Q) + '\n' + 'sigma_px = ' + str(sigma_px) + ' sigma_py =' + str(sigma_py),
                 fontsize=14)
    plt.plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
    plt.plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
    plt.scatter(all_obs[0], all_obs[1], color='red', label='Observations', alpha=0.2)
    plt.legend()
    plt.show()


def question_7_new_mes():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)
    all_P = np.zeros((n, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)
    all_K = np.zeros((n, 4, 2))

    for i in range(1, n):
        [result_matrix[i], all_P[i], all_K[i]] = filtre_de_kalman(
            F, Q, H, R,
            np.transpose(all_obs)[i],
            result_matrix[i - 1],
            all_P[i - 1]
        )

    final_result_matrix = np.transpose(result_matrix)

    #print(" Valeurs pour les prédictions : ")
    #print(final_result_matrix.transpose()[:10])
    #print('Valeurs des positions exactes')
    #print(all_evol.transpose()[:10])
    #print("Valeurs pour les K : ")
    #print(all_K[:10])

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Erreur moyenne  : \n ' +
                 " | ".join(map(str, avg_error_sep(final_result_matrix, all_evol))) + '\n' +
                 'sigma_Q = ' + str(sigma_Q) + ', sigma_px = ' + str(sigma_px) + ', sigma_py =' + str(sigma_py),
                 fontsize=15)

    #print("Erreur séparée : ")
    #print(avg_error_sep(final_result_matrix, all_evol))
    #print('Plus grosse erreur : ')
    #print(sup_errr(final_result_matrix, all_evol))

    plt.plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
    plt.plot(all_evol[0], all_evol[2], color='orange', label='v_orig', alpha=0.7)
    plt.scatter(all_obs[0], all_obs[1], color='red', label='Observations', alpha=0.2)
    plt.legend()
    plt.show()


def question9():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    result_matrix[0] = np.transpose(x_init)

    all_P = np.zeros((n, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')
    all_evol = creer_trajectoire(F, Q, x_kalm, n)
    all_obs = creer_observation(H, R, all_evol)
    all_K = np.zeros((n, 4, 2))

    for i in range(1, n):
        [result_matrix[i], all_P[i], all_K[i]] = filtre_de_kalman(
            F, Q, H, R,
            np.transpose(all_obs)[i],
            result_matrix[i-1],
            all_P[i-1]
        )

    final_result_matrix = np.transpose(result_matrix)
    fig, axs = plt.subplots(5, 1, figsize=(16, 9))
    fig.suptitle('Erreur moyenne sur la position ' + str(avg_erro_vect(final_result_matrix, all_evol)), fontsize=18)

    axs[0].plot(all_evol[0], 'r', label='x_real', alpha=0.7)
    axs[0].plot(final_result_matrix[0], label='x_est', alpha=0.7)
    axs[0].set_title('Suivi de trajectoire sur x', fontsize=16)

    axs[1].plot(all_evol[2], 'r', label='y_real', alpha=0.7)
    axs[1].plot(final_result_matrix[2], label='y_est', alpha=0.7)
    axs[1].set_title('Suivi de trajectoire sur y', fontsize=16)

    axs[2].plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='Position estimée', alpha=0.7)
    axs[2].plot(all_evol[0], all_evol[2], color='orange', label='Position réelle', alpha=0.7)
    axs[2].scatter(all_obs[0], all_obs[1], color='red', label='Position mesurée', alpha=0.4)
    axs[2].set_title('Suivi de la position', fontsize=16)

    K_norm = []
    P_norm = []
    for i in range(len(all_K)):
        K_norm.append(np.linalg.norm(all_K[i]))
        P_norm.append(np.linalg.norm(all_P[i]))
    axs[3].plot(K_norm, color='blue', label='K norm')
    axs[3].set_title("Evolution de la norme de K")

    axs[4].plot(P_norm, color='orange', label='P_norm')
    axs[4].set_title("Evolution de la norme de P")

    for ax in axs.flat:
        ax.legend()

    fig.tight_layout()
    fig.show()

# ----- Partie Application -----


def question3_avec_norme():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    all_P = np.zeros((n, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')
    all_K = np.zeros((n, 4, 2))

    result_matrix[0] = np.transpose(x_init)

    for name in ['ligne', 'voltige']:

        reach = False
        all_evol = scipy.io.loadmat('fichiers_donnees/vecteur_x_avion_' + name + '.mat')['vecteur_x']
        all_obs = scipy.io.loadmat('fichiers_donnees/vecteur_y_avion_' + name + '.mat')['vecteur_y']

        all_obs_available = [np.transpose(all_obs)[0]]
        for i in range(1, n): #Utilisation de 'all_obs_available' dans le cas où plusieurs observations qui se suivent ne sont pas disponibles
            all_obs_available.append(np.transpose(all_obs)[i])
            [result_matrix[i], all_P[i], all_K[i]] = filtre_de_kalman_with_gap_last_observed(F, Q, H, R,
                                                                                             all_obs_available,
                                                                                             result_matrix[i - 1],
                                                                                             all_P[i - 1])

            #Test sur l'impact de la vitesse de convergence de en fonction du type de situation
            if (np.round(all_K[i], 10) == np.round(all_K[i-1], 10)).all() and (not reach):
                print('Convergence atteine en ' + str(i) + " pour " + str(name))
                reach = not reach


        final_result_matrix = np.transpose(result_matrix)
        #print("Valeurs de K " + name + ' : ')
        #print(all_K[5:15])
        print('Erreurs moyennes séparées ' + str(name) + " : ")
        print(avg_error_sep(all_evol, final_result_matrix))
        print('Erreurs maximales séparées ' + str(name) + " : ")
        print(sup_errr(all_evol, final_result_matrix))

        fig, axs = plt.subplots(3, 1, figsize=(16, 9))
        fig.suptitle('Erreur vectorielle moyenne cas : ' + name + ' ' + str(avg_error_sep(final_result_matrix, all_evol)), fontsize=18)

        axs[0].plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
        axs[0].plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
        axs[0].set_title('Comparaison des deux trajectoires')

        K_norm = []
        P_norm = []
        for i in range(len(all_K)):
            K_norm.append(np.linalg.norm(all_K[i]))
            P_norm.append(np.linalg.norm(all_P[i]))
        axs[1].plot(K_norm, color='blue', label='K norm')
        axs[1].set_title("Evolution de la norme de K")

        axs[2].plot(P_norm, color='orange', label='P_norm')
        axs[2].set_title("Evolution de la norme de P")

        for ax in axs.flat:
            ax.legend()

        fig.tight_layout()
        fig.show()


def question_3():
    n = 100
    result_matrix = np.zeros((n, 4), dtype='float64')
    all_P = np.zeros((n, P_kaml.shape[0], P_kaml.shape[1]), dtype='float64')
    all_K = np.zeros((n, 4, 2))

    result_matrix[0] = np.transpose(x_init)

    for name in ['ligne', 'voltige']:

        reach = False
        all_evol = scipy.io.loadmat('fichiers_donnees/vecteur_x_avion_' + name + '.mat')['vecteur_x']
        all_obs = scipy.io.loadmat('fichiers_donnees/vecteur_y_avion_' + name + '.mat')['vecteur_y']
        print(all_evol.shape)

        all_obs_available = [np.transpose(all_obs)[0]]
        for i in range(1,
                       n):  # Utilisation de 'all_obs_available' dans le cas où plusieurs observations qui se suivent ne sont pas disponibles
            all_obs_available.append(np.transpose(all_obs)[i])
            [result_matrix[i], all_P[i], all_K[i]] = filtre_de_kalman_with_gap_pred_only(F, Q, H, R,
                                                                                             all_obs_available[i],
                                                                                             result_matrix[i - 1],
                                                                                             all_P[i - 1])

            # Test sur l'impact de la vitesse de convergence de en fonction du type de situation
            if (np.round(all_K[i], 10) == np.round(all_K[i - 1], 10)).all() and (not reach):
                print('Convergence atteine en ' + str(i) + " pour " + str(name))
                reach = not reach

        final_result_matrix = np.transpose(result_matrix)
        print('Erreurs moyennes séparées ' + str(name) + " : ")
        print(avg_error_sep(all_evol, final_result_matrix))
        print('Erreurs maximales séparées ' + str(name) + " : ")
        print(sup_errr(all_evol, final_result_matrix))

        fig = plt.figure(figsize=(12, 7))
        fig.suptitle("Erreurs moyennes : \n" +
                     ", ".join(map(str, np.round(avg_error_sep(final_result_matrix, all_evol), 3))) + '\n'
                     'Cas : ' + name, fontsize=16)

        plt.plot(final_result_matrix[0], final_result_matrix[2], color='blue', label='X_est', alpha=0.7)
        plt.plot(all_evol[0], all_evol[2], color='orange', label='X_orig', alpha=0.7)
        plt.scatter(all_obs[0], all_obs[1], color='red', label='Observations', alpha=0.3)

        plt.legend()
        plt.show()


#question4()
#question7()
#question_7_new_mes()
#question9()
question_3()

"""fig = plt.figure()
vecteur_x = creer_trajectoire(F, Q, x_init, 100)
vecteur_obs = creer_observation(H, R, vecteur_x)
plt.plot(vecteur_x[0], vecteur_x[2], label='Trajectoire réelle', alpha=0.7)
plt.scatter(vecteur_obs[0], vecteur_obs[1], label='Trajectoire bruitée', alpha=0.4, color='orange')
plt.legend()
plt.show()
"""





