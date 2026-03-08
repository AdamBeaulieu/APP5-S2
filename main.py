# APP5
# Modélisation cinématique d'un systeme robotisé 6 axes d'inspection par vision
# Mora8079 & beaa1691

## importation des bibliothèques nécessaires
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
np.set_printoptions(precision=6, suppress=True)
## initialisation des fonctions
# fonctions trigonométriques
def s(theta):
    """ 
    Calcule le sinus de l'angle theta (en radians) 
    theta : angle de rotation en radians
    """
    return np.sin(theta)

def c(theta):
    """ 
    Calcule le cosinus de l'angle theta (en radians) 
    theta : angle de rotation en radians
    """
    return np.cos(theta)

# rotation des bases de chaque joint
def mat_rot_ang(theta, axe):
    """ 
    Calcule la matrice de rotation autour de l'axe spécifié par l'angle theta (en radians) 
    theta : angle de rotation en radians
    axe : axe de rotation (1 pour x, 2 pour y, 3 pour z) 
    """
    matrice_rotation = np.zeros((3, 3))
    if axe == 1:
        matrice_rotation[0,0] = 1
        matrice_rotation[1,1] = c(theta)
        matrice_rotation[1,2] = -s(theta)
        matrice_rotation[2,1] = s(theta)
        matrice_rotation[2,2] = c(theta)
    elif axe == 2:
        matrice_rotation[0,0] = c(theta)
        matrice_rotation[0,2] = s(theta)
        matrice_rotation[1,1] = 1
        matrice_rotation[2,0] = -s(theta)
        matrice_rotation[2,2] = c(theta)
    elif axe == 3:
        matrice_rotation[0,0] = c(theta)
        matrice_rotation[0,1] = -s(theta)
        matrice_rotation[1,0] = s(theta)
        matrice_rotation[1,1] = c(theta)
        matrice_rotation[2,2] = 1

    return matrice_rotation
# fonction dévaluation des défauts
def eval_defaut(defaut):
    x = defaut[0]
    y = defaut[1]
    return (((45*x**2) + (30*x*y) + (85*y**2) - (10.8*x) - (8.4*y) + 0.684) < 0)
# fonction de transformation de coordonnées camera vers piece
def V2P(rv_V, pRt, tRw, rw_VW, rw_TW, rt_TP):
    rw_V = rv_V # point par rapport à la caméra dans le repere monde
    rw_W = rw_V + rw_VW # point par rapport au monde dans le repere monde
    rt_T = tRw @ (rw_W - rw_TW) # point par rapport au tool dans le repere tool
    rp_P = pRt @ (rt_T + rt_TP) # point par rapport à la piece dans le repere piece
    return rp_P

## initialisation des données
PI = float(np.pi)  # 3.141592653
L1= float(0.15)
L2x = float(0.05)
L2y = float(0.1)
L3 = float(0.5)
L4x = float(0.1)
L4y = float(0.02)
L5 = float(0.3)
L6 = float(0.02)
# pour tout les matrice, premiere rangée: x(1), deuxieme rangée: y(2), troisieme rangée: z(3)
Wo = np.array([[0, 0, 0]]).T# origine du repere world
rw_AW = np.array([[0, L1, 0]]).T # vecteur de W à A (w1, w2, w3)
ra_BA = np.array([[L2x, L2y, 0]]).T # vecteur de A à B (a1, a2, a3)
rb_CB = np.array([[0, L3, 0]]).T # vecteur de B à C (b1, b2, b3)
rc_DC = np.array([[L4x, L4y, 0]]).T # vecteur de C à D (c1, c2, c3)
rd_ED = np.array([[L5, 0, 0]]).T # vecteur de D à E (d1, d2, d3)
re_TE = np.array([[L6, 0, 0]]).T # vecteur de E à T (e1, e2, e3)
# piece
rw_PW = np.array([[0.5994, 0, 0.1991]]).T # position de la piece dans le repere monde
Po = np.array([[0, 0, 0]]).T # position de la piece dans le repere monde
rp_LbP = np.array([[0.15, 0, 0]]).T # longueur de la piece
rp_HgP = np.array([[0, 0.1, 0]]).T # grande hauteur de la piece
rp_HdP = np.array([[0.15, 0.05, 0]]).T # petite hauteur de la piece
# caméra
rw_VW = np.array([[0.8, 0.7, 0]]).T # position de la caméra dans le repere monde
# point de la tranche (repere camera)
TI_v = np.array([[0.158920, 0.157470, 0.153781, 0.152420, 0.150931],
             [0.013914, 0.021067, 0.039266, 0.045970, 0.053326],
             [0.028686, 0.008891, -0.040587, -0.060395, -0.080185]])
# défaut de la piece (repere caméra)
DF_v = np.array([[0.153758, 0.145698, 0.153932, 0.152097, 0.146104],
               [0.039379, 0.079138, 0.038521, 0.047573, 0.077134],
               [-0.025575, -0.039398, 0.009411, 0.035692, 0.030571]])

#######################################
########## DÉBUT DU CODE ##############
#######################################
# rw_PW = mat_rot_ang(PI/2, 1) @ rw_PW
# rw_LbP = mat_rot_ang(PI/2, 1) @ rp_LbP
# rw_HgP = mat_rot_ang(PI/2, 1) @ rp_HgP
# rw_HdP = mat_rot_ang(PI/2, 1) @ rp_HdP

#######################################
# validation de la configuration de départ
#######################################
qT_valid = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0]) # configuration de validation
# matrice de rotation pour chaque joint
wRa = mat_rot_ang(qT_valid[0], 2)
aRb = mat_rot_ang(qT_valid[1], 3)
bRc = mat_rot_ang(qT_valid[2], 3)
cRd = mat_rot_ang(qT_valid[3], 1)
dRe = mat_rot_ang(qT_valid[4], 3)
eRt = mat_rot_ang(qT_valid[5], 1)
# matrice de rotation succecive
wRb = wRa @ aRb
wRc = wRb @ bRc
wRd = wRc @ cRd
wRe = wRd @ dRe
wRt = wRe @ eRt
# calcul des positions des points de référence dans le repere monde
rw_BW = (wRa @ ra_BA) + rw_AW
rw_CW = (wRb @ rb_CB) + rw_BW
rw_DW = (wRc @ rc_DC) + rw_CW
rw_EW = (wRd @ rd_ED) + rw_DW
rw_TW = (wRe @ re_TE) + rw_EW
print("\n")
print("##############################################")
print("VALIDATIION DU PROGRAMME\n")
print("Position de l'effecteur lorsque tout les angle sont à 0.1 rad: \n", rw_TW)
#######################################
# Prise de la piece
#######################################
qT_prise = np.array([-0.4, -1.2, 0, 0, -0.3708, 0]) # prise de la pièce
# matrice de rotation pour chaque joint
wRa = mat_rot_ang(qT_prise[0], 2)
aRb = mat_rot_ang(qT_prise[1], 3)
bRc = mat_rot_ang(qT_prise[2], 3)
cRd = mat_rot_ang(qT_prise[3], 1)
dRe = mat_rot_ang(qT_prise[4], 3)
eRt = mat_rot_ang(qT_prise[5], 1)
# matrice de rotation succecive
wRb = wRa @ aRb
wRc = wRb @ bRc
wRd = wRc @ cRd
wRe = wRd @ dRe
wRt = wRe @ eRt
# calcul des positions des points de référence dans le repere monde
rw_BW = (wRa @ ra_BA) + rw_AW
rw_CW = (wRb @ rb_CB) + rw_BW
rw_DW = (wRc @ rc_DC) + rw_CW
rw_EW = (wRd @ rd_ED) + rw_DW
rw_TW = (wRe @ re_TE) + rw_EW
rw_PT = -1 * (rw_TW - rw_PW) # vecteur de P vers T dans le repere monde
rt_PT = wRt.T @ rw_PT # vecteur de P vers T dans le repere tool

#######################################
# Inspection de la piece
#######################################
qT_insp = np.array([0, -0.3, 0, 0, 0.5, -1.6]) # inspection de la face avant de la pièce
# matrice de rotation pour chaque joint
wRa = mat_rot_ang(qT_insp[0], 2)
aRb = mat_rot_ang(qT_insp[1], 3)
bRc = mat_rot_ang(qT_insp[2], 3)
cRd = mat_rot_ang(qT_insp[3], 1)
dRe = mat_rot_ang(qT_insp[4], 3)
eRt = mat_rot_ang(qT_insp[5], 1)
# matrice de rotation succecive
wRb = wRa @ aRb
wRc = wRb @ bRc
wRd = wRc @ cRd
wRe = wRd @ dRe
wRt = wRe @ eRt
# calcul des positions des points de référence dans le repere monde
rw_BW = (wRa @ ra_BA) + rw_AW
rw_CW = (wRb @ rb_CB) + rw_BW
rw_DW = (wRc @ rc_DC) + rw_CW
rw_EW = (wRd @ rd_ED) + rw_DW
rw_TW = (wRe @ re_TE) + rw_EW
# enregistrement des positions
Robot_insp = np.array([Wo, rw_AW, rw_BW, rw_CW, rw_DW, rw_EW, rw_TW])
To_inspection = rw_TW
# print("\nPosition de l'effecteur :\n", To_inspection) 

#######################################
# Analyse des défauts et de la tranche
#######################################
wRv = np.array([[-1, 0, 0],
               [0, -1, 0],
               [0, 0, 1]]) # matrice de rotation de V vers W
tRw = wRt.T # matrice de rotation de W vers T
pRt=  np.array([[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]) # matrice de rotation de T vers P 
rt_TP = -1 * rt_PT # vecteur de P vers T dans le repere tool
# Vecteur x, y pour le tracé de la zone interdite et de la droite d'approximation de la tranche
x = np.linspace(0, 0.15, 150)
y = np.linspace(0, 0.15, 150)
## Défauts
DF_w = wRv @ DF_v # défaut de la piece dans le repere monde
print("\n")
print("##############################################")
print("DÉFAUT\n")
print("Des défauts ont été détectés, voici leur position par rapport à l'origine de la pièce:")
# defaut 1
rp_Df1P = V2P(np.array([DF_w[:, 0]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
print("Defaut 1:\n", rp_Df1P, "\n", "Dans la zone critique: ", eval_defaut(rp_Df1P))
# defaut 2
rp_Df2P = V2P(np.array([DF_w[:, 1]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
print("Defaut 2:\n", rp_Df2P, "\n", "Dans la zone critique: ", eval_defaut(rp_Df2P))
#defaut 3
rp_Df3P = V2P(np.array([DF_w[:, 2]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
print("Defaut 3:\n", rp_Df3P, "\n", "Dans la zone critique: ", eval_defaut(rp_Df3P))
#defaut 4
rp_Df4P = V2P(np.array([DF_w[:, 3]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
print("Defaut 4:\n", rp_Df4P, "\n", "Dans la zone critique: ", eval_defaut(rp_Df4P))
#defaut 5
rp_Df5P = V2P(np.array([DF_w[:, 4]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
print("Defaut 5:\n", rp_Df5P, "\n", "Dans la zone critique: ", eval_defaut(rp_Df5P))
# Zone interdite
X, Y = np.meshgrid(x, y)
Zone = 45*X**2 + 30*X*Y + 85*Y**2 - 10.8*X - 8.4*Y + 0.684
## Tranche
TI_w = wRv @ TI_v # tranche de la piece dans le repere monde
rp_Ti1P = V2P(np.array([TI_w[:, 0]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
rp_Ti2P = V2P(np.array([TI_w[:, 1]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
rp_Ti3P = V2P(np.array([TI_w[:, 2]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
rp_Ti4P = V2P(np.array([TI_w[:, 3]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
rp_Ti5P = V2P(np.array([TI_w[:, 4]]).T, pRt, tRw, rw_VW, rw_TW, rt_TP)
TI_p = np.column_stack((rp_Ti1P, rp_Ti2P, rp_Ti3P, rp_Ti4P, rp_Ti5P)).T
TI_x = np.array(TI_p[:, 0])
TI_y = np.array(TI_p[:, 1])
one = np.ones_like(TI_x)
A = np.column_stack((TI_x, one))
X_approx = np.linalg.inv(A.T @ A) @ A.T @ TI_y
# droite d'approximation de la tranche
droite_tranche = X_approx[0] * x + X_approx[1]
dir_tranche = np.array([(x[75]- x[0]), (droite_tranche[75] - droite_tranche[0])])
dir_tranche = dir_tranche / np.linalg.norm(dir_tranche)
# Calcul de l'angle phi (angle entre la tranche et p1)
phi = np.arctan2(dir_tranche[1], dir_tranche[0])
phi = abs(phi * 180 / PI)
# print("Direction de la tranche: \n", dir_tranche)
print("\n")
print("##############################################")
print("TRANCHE\n")
print("Distance de départ de la tranche:", round(droite_tranche[0], 3), "m")
print("Angle phi (entre la tranche et p1): ", round(phi, 1), "degrée")


#######################################
# Selon angle donné
#######################################


## Modifier les angles ci-bas au besoin
# qT_entre = np.array([0, 0, 1.521, 0, 0, 0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta
qT_entre = np.array([-0.4, -1.2, 0, 0, -0.3708, 0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta

# matrice de rotation pour chaque joint
wRa = mat_rot_ang(qT_entre[0], 2)
aRb = mat_rot_ang(qT_entre[1], 3)
bRc = mat_rot_ang(qT_entre[2], 3)
cRd = mat_rot_ang(qT_entre[3], 1)
dRe = mat_rot_ang(qT_entre[4], 3)
eRt = mat_rot_ang(qT_entre[5], 1)
# matrice de rotation succecive
wRb = wRa @ aRb
wRc = wRb @ bRc
wRd = wRc @ cRd
wRe = wRd @ dRe
wRt = wRe @ eRt
# calcul des positions des points de référence dans le repere monde
rw_BW = (wRa @ ra_BA) + rw_AW
rw_CW = (wRb @ rb_CB) + rw_BW
rw_DW = (wRc @ rc_DC) + rw_CW
rw_EW = (wRd @ rd_ED) + rw_DW
rw_TW = (wRe @ re_TE) + rw_EW
# enregistrement des positions
Robot_entre = np.array([Wo, rw_AW, rw_BW, rw_CW, rw_DW, rw_EW, rw_TW])
To_entre = rw_TW
print("\n")
print("##############################################")
print("POSITION SELON LES ANGLES ENTRÉS\n")
print("Les angles entrés sont: ", qT_entre)
print("La position de l'effecteur selon l'angle entré:\n", To_entre)

#######################################
# Cinématique différentielle
#######################################
a1 = -(L3*s(qT_entre[1])) - (L4y*s(qT_entre[1]+qT_entre[2])) + ((L4x+L5)*c(qT_entre[1]+qT_entre[2]))
a2 = - (L4y*s(qT_entre[1]+qT_entre[2])) + ((L4x+L5)*c(qT_entre[1]+qT_entre[2]))
h_dot = np.array([[1]]) # Initialisation de la vitesse verticale désirée pour le point Eo.
# scenario a
Jh_a = np.array([[0, a1, a2, 0, 0, 0]])
q_dot_a = Jh_a.T @ (np.linalg.inv(Jh_a @ Jh_a.T)) @ h_dot
# scenario b
Jh_b = np.array([[0, a1, a2, 0, 0, 0]])
q_dot_b = Jh_b.T @ (np.linalg.inv(Jh_b @ Jh_b.T)) @ h_dot
# scenario c
Jh_c = np.array([[0, a1, 0, 0, 0, 0]])
q_dot_c = Jh_c.T @ (np.linalg.inv(Jh_c @ Jh_c.T)) @ h_dot
print("\n")
print("##############################################\n")
print("CINÉMATIQUE DIFFÉRENTIELLE\n")
print("La vitesse angulaire des joints doivent être:")
print("   - Pour le scenario a)\n", q_dot_a)
print("   - Pour le scenario b)\n", q_dot_b)
print("   - Pour le scenario c)\n", q_dot_c)
print("\n")
print("##############################################\n")

#######################################
# Affichage
#######################################
## Afficher le robot en position d'inspection
insp = plt.figure("Robot à l'inspection")
ax0 = insp.add_subplot(111, projection='3d')  
ax0.set_xlabel('W1')
ax0.set_ylabel('W3')
ax0.set_zlabel('W2')
ax0.invert_yaxis()
ax0.set_xlim(-1, 1)
ax0.set_ylim(-1, 1)
#dessiner les membres du robot
Robot_insp_w1 = np.array(Robot_insp[:,0])
Robot_insp_w3 = np.array(Robot_insp[:,2])
Robot_insp_w2 = np.array(Robot_insp[:,1])
# affichage de la configuration du robot
ax0.plot(Robot_insp_w1, Robot_insp_w3, Robot_insp_w2, 'r.-', label = "Robot") # robot
#afficher les# points de référence
ax0.plot([Wo[0]],[Wo[2]], [Wo[1]], 'bo-', label = "World") # base du robot
ax0.plot([To_inspection[0]],[To_inspection[2]], [To_inspection[1]], 'co-', label = "Tool") # position de l'outil
ax0.plot([rw_VW[0]],[rw_VW[2]], [rw_VW[1]], 'mo-', label = "Camera") # position de la caméra
## Afficher le robot selon les angles entrés
entre = plt.figure("Robot selon les angles entrés")
ax1 = entre.add_subplot(111, projection='3d')  
ax1.set_xlabel('W1')
ax1.set_ylabel('W3')
ax1.set_zlabel('W2')
ax1.invert_yaxis()
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
#dessiner les membres du robot
Robot_entre_w1 = np.array(Robot_entre[:,0])
Robot_entre_w3 = np.array(Robot_entre[:,2])
Robot_entre_w2 = np.array(Robot_entre[:,1])
# affichage de la configuration du robot
ax1.plot(Robot_entre_w1, Robot_entre_w3, Robot_entre_w2, 'r.-', label = "Robot") # robot
#afficher les# points de référence
ax1.plot([Wo[0]],[Wo[2]], [Wo[1]], 'bo-', label = "World") # base du robot
ax1.plot([To_entre[0]],[To_entre[2]], [To_entre[1]], 'co-', label = "Tool") # position de l'outil
ax1.plot([rw_VW[0]],[rw_VW[2]], [rw_VW[1]], 'mo-', label = "Camera") # position de la caméra
## Afficher les défauts
defaut = plt.figure("Défauts détectés")
ax2 = defaut.add_subplot()
ax2.set_xlim(-0.01, 0.175)
ax2.set_ylim(-0.01, 0.175)
Piece_P = np.array([Po, rp_LbP, rp_HdP, rp_HgP, Po])
Piece_p1 = np.array(Piece_P[:,0])
Piece_p2 = np.array(Piece_P[:,1])
Piece_p3 = np.array(Piece_P[:,2])
ax2.contour(X, Y, Zone, levels=[0], colors='m', linestyles='dashed') # zone interdite
ax2.plot([rp_Df1P[0]],[rp_Df1P[1]], 'ro', label = "DF1") # défaut 1
ax2.plot([rp_Df2P[0]],[rp_Df2P[1]], 'yo', label = "DF2") # défaut 2
ax2.plot([rp_Df3P[0]],[rp_Df3P[1]], 'bo', label = "DF3") # défaut 3
ax2.plot([rp_Df4P[0]],[rp_Df4P[1]], 'co', label = "DF4") # défaut 4
ax2.plot([rp_Df5P[0]],[rp_Df5P[1]], 'mo', label = "DF5") # défaut 5
ax2.plot(Piece_p1, Piece_p2, 'k', label = "Piece") # pièce à inspecter
ax2.plot(TI_x, TI_y, 'go', label = "Tranche") # pièce à inspecter
ax2.plot(x, droite_tranche, 'r--', label = "Approximation de la tranche") # droite d'approximation de la tranche
plt.legend()
plt.show()