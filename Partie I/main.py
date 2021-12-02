import numpy as np


def creer_trajectoire(F, Q, x_init, T): #Format 4*n
    trajectoire = [np.transpose(x_init)]
    for i in range(1, T):
        trajectoire.append(
            F.dot(trajectoire[-1]) +
            np.random.multivariate_normal(np.zeros((4, 1), dtype='float64').flatten(), Q)
        )
    return np.transpose(trajectoire)


def creer_observation(H, R, vecteur_x): #Format 2*n
    return(
        H.dot(vecteur_x) + np.transpose(np.random.multivariate_normal(np.zeros((2, 1), dtype='float64').flatten(), R, len(vecteur_x[0])))
    )


def filtre_de_kalman(F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):

    #Partie prédiction :
    x_pred_suiv = F.dot(x_kalm_prec)
    P_pred_suiv = F.dot(P_kalm_prec.dot(np.transpose(F))) + Q

    #Partie filtrage :
    inv_matrix = np.linalg.inv(H.dot(P_pred_suiv.dot(np.transpose(H))) + R)
    K_filt_suiv = P_pred_suiv.dot(np.transpose(H).dot(inv_matrix))
    P_filt_suiv = (np.eye(4, dtype='float64') - K_filt_suiv.dot(H)).dot(P_pred_suiv)
    x_filt_suiv = x_pred_suiv + K_filt_suiv.dot(y_k - H.dot(x_pred_suiv))

    return [x_filt_suiv, P_filt_suiv]


#Fonction intermédiaire pour faciliter les calculs :


def get_P_and_K(P_init, H, Q, R, F, n):
    results = [[P_init], [P_init.dot(np.transpose(H).dot(np.linalg.inv(H.dot(P_init.dot(np.transpose(H))) + R)))]]

    for i in range(1, n):
        results[0].append(F.dot(results[0][-1].dot(np.transpose(F))) + Q)
        results[1].append(
            results[0][-1].dot(np.transpose(H).dot(np.linalg.inv(H.dot(results[0][-1].dot(np.transpose(H))) + R)))
        )
    return results


def err_quadra(x_real, x_est): #Erreur non rapportée à X (seulement un produit) -> Comment pourrait-elle être interprétable ?
    return (x_real - x_est).dot(np.transpose(x_real - x_est))
    #Critique sur le format de l'erreur calculée si on applique directement la formule donnée par l'énonce (on fait
    #Des erreur sur les vitesses, mais aussi on compare la vitesse estimée et la position réelle .. bref
    #Et sur la dimension de l'erreur : c'est une position, or X varie de 0 à 8000 donc l'interprétab.. (voir début)


def average_err(X_real, X_est): #Format 4*n
    return np.array([[np.sqrt(err_quadra(X_real[0], X_est[0])), np.sqrt(err_quadra(X_real[2], X_est[2]))]])/len(X_real[0])











