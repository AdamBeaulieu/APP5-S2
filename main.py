# APP5
# Modélisation cinématique d'un systeme robotisé 6 axes d'inspection par vision

## importation des bibliothèques nécessaires
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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



# matrice de rotation à partir de deux bases
# def mat_rot_base(base1, base2):
#     """
#     Calcule la matrice de rotation pour passer de la base1 à la base2
#     base1 : base de référence actuelle
#     base2 : base de référence souhaitée
#     """
#     rotation = np.zeros((3, 3))
#     for i in range(3):
#         for j in range(3):
#             rotation[i, j] = np.dot(base1[i], base2[j])
#     return rotation    

# changement de base
# def changement_base(vecteur, R):
#     """ 
#     Effectue un changement de base du vecteur de la base1 à la base2 
#     vecteur : vecteur à transformer
#     R : matrice de rotation pour passer de la base1 à la base2
#     """
#     new_vector =  R@vecteur
#     return new_vector

# # mise a jour des positions
# def MaJ_pos(Vec_pos, base1, base2, theta, axe):
#     """
#     mets à jour la position du vecteur en effectuant une rotation et un changement de base
#     Vec_pos : position du vecteur à mettre à jour
#     base1 : base de référence actuelle du vecteur
#     base2 : base de référence souhaitée du vecteur
#     theta : angle de rotation en radians
#     axe : axe de rotation (1 pour x, 2 pour y, 3 pour z)
#     """
#     # new_base = rotation(base1,theta,axe)
#     # new_pos = changement_base(Vec_pos, new_base, base2)
#     rotation_mat = mat_rotation(theta, axe)
#     new_pos = np.dot(rotation_mat, Vec_pos)
#     return new_pos

## initialisation des données
PI = float(np.pi)  # 3.141592653
# pour tout les matrice, premiere rangée: x(1), deuxieme rangée: y(2), troisieme rangée: z(3)
qT_zero = np.array([0, 0, 0, 0, 0, 0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta
qT_insp = np.array([0, -0.3, 0, 0, 0.5, -1.6]) # inspection de la face avant de la pièce
qT_prise = np.array([-0.4, -1.2, 0, 0, -0.3708, 0]) # prise de la pièce
qT_valid = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0]) # configuration de validation

Wo = np.array([[0, 0, 0]]).T# origine du repere world
rw_AW = np.array([[0, 0.15, 0]]).T # vecteur de W à A (w1, w2, w3)
ra_BA = np.array([[0.05, 0.1, 0]]).T # vecteur de A à B (a1, a2, a3)
rb_CB = np.array([[0, 0.5, 0]]).T # vecteur de B à C (b1, b2, b3)
rc_DC = np.array([[0.1, 0.02, 0]]).T # vecteur de C à D (c1, c2, c3)
rd_ED = np.array([[0.3, 0, 0]]).T # vecteur de D à E (d1, d2, d3)
re_TE = np.array([[0.02, 0, 0]]).T # vecteur de E à T (e1, e2, e3)

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
rw_LbP = mat_rot_ang(PI/2, 1) @ rp_LbP
rw_HgP = mat_rot_ang(PI/2, 1) @ rp_HgP
rw_HdP = mat_rot_ang(PI/2, 1) @ rp_HdP


#######################################
# validation de la configuration de départ
#######################################
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

print("validation tool/world: \n", rw_TW)

#######################################
# Prise de la piece
#######################################

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
rw_TP = rw_TW - rw_PW # vecteur de P vers T dans le repere monde
print("tool/piece: \n", rw_TP)
print("prise tool/world: \n", rw_TW)

#######################################
# Inspection de la piece
#######################################
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

print("inspection tool/world: \n", rw_TW)

## calcul des défaut selon le repere piece
# Rotation du repere camera vers repere monde
wRv = np.array([[-1, 0, 0],
               [0, -1, 0],
               [0, 0, 1]]) # matrice de rotation de V vers W
DF_w = wRv @ DF_v # défaut de la piece dans le repere monde
wRt = wRa @ aRb @ bRc @ cRd @ dRe @ eRt
tRw = wRt.T # matrice de rotation de W vers T
# rotation du tool vers la piece
pRt=  np.array([[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]) # matrice de rotation de T vers P 
rt_TP = tRw @ rw_TP # vecteur de P vers T dans le repere tool

# defaut 1
rw_Df1V = np.array([DF_w[:, 0]]).T # defaut1 par rapport à la caméra dans le repere monde
# print("Defaut 1 cam rotationé: \n", rw_Df1V)
rw_Df1W = rw_Df1V + rw_VW # defaut1 par rapport au monde dans le repere monde
# print("Defaut 1 world: \n", rw_Df1W)
rt_Df1T = tRw @ (rw_Df1W - rw_TW) # defaut1 par rapport au tool dans le repere tool
# print("Defaut 1 tool: \n", rt_Df1T)
rp_Df1P = pRt @ (rt_Df1T + rt_TP) # defaut1 par rapport à la piece dans le repere piece
print("Defaut 1 piece: \n", rp_Df1P)

# defaut 2
rw_Df2V = np.array([DF_w[:, 1]]).T # defaut2 par rapport à la caméra dans le repere monde
rw_Df2W = rw_Df2V + rw_VW # defaut2 par rapport au monde dans le repere monde
rt_Df2T = tRw @ (rw_Df2W - rw_TW) # defaut2 par rapport au tool dans le repere tool
rp_Df2P = pRt @ (rt_Df2T + rt_TP) # defaut2 par rapport à la piece dans le repere piece
print("Defaut 2 piece: \n", rp_Df2P)
#defaut 3
rw_Df3V = np.array([DF_w[:, 2]]).T # defaut3 par rapport à la caméra dans le repere monde
rw_Df3W = rw_Df3V + rw_VW # defaut3 par rapport au monde dans le repere monde
rt_Df3T = tRw @ (rw_Df3W - rw_TW) # defaut3 par rapport au tool dans le repere tool
rp_Df3P = pRt @ (rt_Df3T + rt_TP) # defaut3 par rapport à la piece dans le repere piece
#defaut 4
rw_Df4V = np.array([DF_w[:, 3]]).T # defaut4 par rapport à la caméra dans le repere monde
rw_Df4W = rw_Df4V + rw_VW # defaut4 par rapport au monde dans le repere monde
rt_Df4T = tRw @ (rw_Df4W - rw_TW) # defaut4 par rapport au tool dans le repere tool
rp_Df4P = pRt @ (rt_Df4T + rt_TP) # defaut4 par rapport à la piece dans le repere piece
#defaut 5
rw_Df5V = np.array([DF_w[:, 4]]).T # defaut5 par rapport à la caméra dans le repere monde
rw_Df5W = rw_Df5V + rw_VW # defaut5 par rapport au monde dans le repere monde
rt_Df5T = tRw @ (rw_Df5W - rw_TW) # defaut5 par rapport au tool dans le repere tool
rp_Df5P = pRt @ (rt_Df5T + rt_TP) # defaut5 par rapport à la piece dans le repere piece




# setup de l'affichage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
ax.set_xlabel('W1')
ax.set_ylabel('W3')
ax.set_zlabel('W2')
ax.invert_yaxis()
plt.xlim(-0.1, 1)
plt.ylim(-0.1, 1)
# dessiner la piece
rw_LbW = rw_PW + rp_LbP
rw_HgW = rw_PW + rp_HgP
rw_HdW = rw_PW + rp_HdP
Piece_W = np.array([rw_PW, rw_LbW, rw_HdW, rw_HgW, rw_PW])
Piece_w1 = np.array([Piece_W[:,0]])
Piece_w3 = np.array([Piece_W[:,2]])
Piece_w2 = np.array([Piece_W[:,1]])
#dessiner les membres du robot
Robot = np.array([Wo, rw_AW, rw_BW, rw_CW, rw_DW, rw_EW, rw_TW])
Robot_w1 = np.array([Robot[:,0]])
Robot_w3 = np.array([Robot[:,2]])
Robot_w2 = np.array([Robot[:,1]])
print("Tool position : ", rw_TW)  

# affichage de la configuration du robot
ax.plot(Robot_w1, Robot_w3, Robot_w2, 'r.-', label = "Robot") # robot
# afficher la pièce à inspecter
ax.plot(Piece_w1, Piece_w3, Piece_w2, 'k.-', label = "Piece") # pièce à inspecter
#afficher les# points de référence
ax.plot([Wo[0]],[Wo[2]], [Wo[1]], 'bo-', label = "World") # base du robot
ax.plot([rw_TW[0]],[rw_TW[2]], [rw_TW[1]], 'co-', label = "Tool") # position de l'outil
ax.plot([rw_VW[0]],[rw_VW[2]], [rw_VW[1]], 'mo-', label = "Camera") # position de la caméra
ax.plot([rw_PW[0]],[rw_PW[2]], [rw_PW[1]], 'go-', label = "Piece") # position de la pièce
# plt.legend()
plt.show()

# afficher les défauts
defaut = plt.figure()
# ax2 = defaut.add_subplot(111, projection='3d')
plt.xlim(-0.01, 0.175)
plt.ylim(-0.01, 0.175)
Piece_P = np.array([Po, rp_LbP, rp_HdP, rp_HgP, Po])
Piece_p1 = np.array([Piece_P[:,0]])
Piece_p3 = np.array([Piece_P[:,2]])
Piece_p2 = np.array([Piece_P[:,1]])
# ax2.plot([rp_Df1P[0]],[rp_Df1P[2]], [rp_Df1P[1]], 'ro-', label = "DF1") # défaut 1
# ax2.plot([rp_Df2P[0]],[rp_Df2P[2]], [rp_Df2P[1]], 'go-', label = "DF2") # défaut 2
# ax2.plot([rp_Df3P[0]],[rp_Df3P[2]], [rp_Df3P[1]], 'bo-', label = "DF3") # défaut 3
# ax2.plot([rp_Df4P[0]],[rp_Df4P[2]], [rp_Df4P[1]], 'co-', label = "DF4") # défaut 4
# ax2.plot([rp_Df5P[0]],[rp_Df5P[2]], [rp_Df5P[1]], 'mo-', label = "DF5") # défaut 5
plt.plot([rp_Df1P[0]],[rp_Df1P[1]],'ro-', label = "DF1") # défaut 1
plt.plot([rp_Df2P[0]],[rp_Df2P[1]],'go-', label = "DF2") # défaut 2
plt.plot([rp_Df3P[0]],[rp_Df3P[1]],'bo-', label = "DF3") # défaut 3
plt.plot([rp_Df4P[0]],[rp_Df4P[1]],'co-', label = "DF4") # défaut 4
plt.plot([rp_Df5P[0]],[rp_Df5P[1]],'mo-', label = "DF5") # défaut 5
plt.plot([0, 0.15, 0.15, 0, 0], [0, 0, 0.05, 0.1, 0], 'k.-', label = "Piece") # pièce à inspecter
# ax2.plot(Piece_p1, Piece_p3, Piece_p2, 'k.-', label = "Piece") # pièce à inspecter



plt.show()



