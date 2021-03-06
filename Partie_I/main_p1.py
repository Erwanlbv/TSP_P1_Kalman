import numpy as np


def creer_trajectoire(F, Q, x_init, T): #Format 4*n
    trajectoire = np.zeros((100, 4), dtype='float64')
    trajectoire[0] = x_init
    for i in range(1, T):
        trajectoire[i] = F.dot(trajectoire[i-1]) + np.random.multivariate_normal(np.zeros(4), Q)
    return trajectoire.transpose()


def creer_observation(H, R, vecteur_x): #Format 2*n
    return H.dot(vecteur_x) + np.transpose(np.random.multivariate_normal(np.zeros(2), R, len(vecteur_x[0])))


def err_quadra_sep(x_real, x_est): #Format 4*1
    err = 0
    err_pos = 0
    err_vit = 0

    for i in range(len(x_real)):
        a = (x_real[i] - x_est[i]).dot(np.transpose(x_real[i] - x_est[i]))
        if (i == 0) or (i == 2):
            err_pos += a
        else:
            err_vit += a
        err += a
        #print(a/100)
    return err, err_pos, err_vit
    #Critique sur le format de l'erreur calculée si on applique directement la formule donnée par l'énonce (on fait
    #Des erreur sur les vitesses, mais aussi on compare la vitesse estimée et la position réelle .. bref
    #Et sur la dimension de l'erreur : c'est une position, or X varie de 0 à 8000 donc l'interprétab.. (voir début)
    #Critique : on somme l'erreur sur la vitesse avec l'erreur sur la position sans passer par des taux donc la
    #validité "physique" est morte, il faut plus voir cette erreur comme une erreur "vectoriel" c'est à dire
    #la distance entre les deux vecteurs.


def err_quadra(x_real, x_est):
    err = 0

    for i in range(len(x_real[0])):
        err += (x_real[:, i] - x_est[:, i]).dot(np.transpose(x_real[:, i] - x_est[:, i]))
    return err


def avg_erro_vect(X_real, X_est): #Format 4*n
    return np.sqrt(err_quadra(X_real, X_est)/len(X_real[0]))


def avg_error_sep(X_real, X_est): #Format 4*n
    return np.sqrt(err_quadra_sep(X_real, X_est))/len(X_real[0])


def sup_errr(X_real, X_est): #Format 4*n
    return np.max(np.abs(X_real[0]-X_est[0])), np.max(np.abs(X_real[2] - X_est[2]))


#Refaire question 8 -> Comparaison en tant que vecteurs purs par vraiment de signification physique
#Pas utile pour la question 10 -> Quelle approche choisir pour avoir une erreur "pertinente" ?
#Introduction un problème de lissage -> ?
#Pour la 10 in utilise l'erreur de la 8

def filtre_de_kalman(F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):

    #Partie prédiction :
    x_pred_suiv = F.dot(x_kalm_prec)
    P_pred_suiv = F.dot(P_kalm_prec.dot(np.transpose(F))) + Q

    #Partie filtrage :
    inv_matrix = np.linalg.inv(H.dot(P_pred_suiv.dot(np.transpose(H))) + R)
    K_filt_suiv = P_pred_suiv.dot(np.transpose(H).dot(inv_matrix))
    P_filt_suiv = (np.eye(4, dtype='float64') - K_filt_suiv.dot(H)).dot(P_pred_suiv)
    x_filt_suiv = x_pred_suiv + K_filt_suiv.dot(y_k - H.dot(x_pred_suiv))

    return [x_filt_suiv, P_filt_suiv, K_filt_suiv]


def filtre_de_kalman_with_gap_last_observed(F, Q, H, R, all_obs_available, x_kalm_prec, P_kalm_prec):

    #Partie prédiction :
    x_pred_suiv = F.dot(x_kalm_prec)
    P_pred_suiv = F.dot(P_kalm_prec.dot(np.transpose(F))) + Q

    #Partie filtrage :
    inv_matrix = np.linalg.inv(H.dot(P_pred_suiv.dot(np.transpose(H))) + R)
    K_filt_suiv = P_pred_suiv.dot(np.transpose(H).dot(inv_matrix))
    P_filt_suiv = (np.eye(4, dtype='float64') - K_filt_suiv.dot(H)).dot(P_pred_suiv)

    #print('Y : ' + str(all_obs_available[-1]))
    #print('Y-pre : ' + str(all_obs_available[-2]))
    i = 1
    while np.isnan(all_obs_available[-i]).any():
        i += 1
    y = all_obs_available[-i]
    #print(' Y final : ' + str(y))
    x_filt_suiv = x_pred_suiv + K_filt_suiv.dot(y - H.dot(x_pred_suiv))

    return [x_filt_suiv, P_filt_suiv, K_filt_suiv]

def filtre_de_kalman_with_gap_pred_only(F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):

    #Partie prédiction :
    x_pred_suiv = F.dot(x_kalm_prec)
    P_pred_suiv = F.dot(P_kalm_prec.dot(np.transpose(F))) + Q

    #Partie filtrage :
    inv_matrix = np.linalg.inv(H.dot(P_pred_suiv.dot(np.transpose(H))) + R)
    K_filt_suiv = P_pred_suiv.dot(np.transpose(H).dot(inv_matrix))
    P_filt_suiv = (np.eye(4, dtype='float64') - K_filt_suiv.dot(H)).dot(P_pred_suiv)

    #print('Y : ' + str(all_obs_available[-1]))
    #print('Y-pre : ' + str(all_obs_available[-2]))
    i = 1
    if np.isnan(y_k).any():
        x_filt_suiv = x_pred_suiv
    else :
        x_filt_suiv = x_pred_suiv + K_filt_suiv.dot(y_k - H.dot(x_pred_suiv))

    return [x_filt_suiv, P_filt_suiv, K_filt_suiv]









