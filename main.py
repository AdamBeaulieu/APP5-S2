# APP5
# Modélisation cinématique d'un systeme robotisé 6 axes d'inspection par vision

## importation des bibliothèques nécessaires
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

## initialisation des fonctions

# rotation des bases de chaque joint
def mat_rot_ang(theta, axe):
    """ 
    CAlcule la matrice de rotation autour de l'axe spécifié par l'angle theta (en radians) 
    theta : angle de rotation en radians
    axe : axe de rotation (1 pour x, 2 pour y, 3 pour z) 
    """
    matrice_rotation = np.zeros((3, 3))
    if axe == 1:
        matrice_rotation[0,0] = 1
        matrice_rotation[1,1] = np.cos(theta)
        matrice_rotation[1,2] = -np.sin(theta)
        matrice_rotation[2,1] = np.sin(theta)
        matrice_rotation[2,2] = np.cos(theta)
    elif axe == 2:
        matrice_rotation[0,0] = np.cos(theta)
        matrice_rotation[0,2] = np.sin(theta)
        matrice_rotation[1,1] = 1
        matrice_rotation[2,0] = -np.sin(theta)
        matrice_rotation[2,2] = np.cos(theta)
    elif axe == 3:
        matrice_rotation[0,0] = np.cos(theta)
        matrice_rotation[0,1] = -np.sin(theta)
        matrice_rotation[1,0] = np.sin(theta)
        matrice_rotation[1,1] = np.cos(theta)
        matrice_rotation[2,2] = 1

    return matrice_rotation

# matrice de rotation à partir de deux bases
def mat_rot_base(base1, base2):
    """
    Calcule la matrice de rotation pour passer de la base1 à la base2
    base1 : base de référence actuelle
    base2 : base de référence souhaitée
    """
    rotation = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rotation[i, j] = np.dot(base1[i], base2[j])
    return rotation    

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
#qT = np.array([0, 0, 0, 0, 0, 0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta
#qT = np.array([0, -0.3, 0, 0, 0.5, -1.6]) # inspection de la face avant de la pièce
#qT = np.array([-0.4, -1.2, 0, 0, -0.3708, 0]) # prise de la pièce
qT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # configuration de validation

Wo = np.array([[0, 0, 0]]).T# origine du repere world
W = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere world
ra_AWo = np.array([[0, 0.15, 0]]).T # vecteur de W à A (a1, a2, a3)
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 2
rb_BA = np.array([[0.05, 0.1, 0]]).T # vecteur de A à B (b1, b2, b3)
B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 3
rc_CB = np.array([[0, 0.5, 0]]).T # vecteur de B à C (c1, c2, c3)
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 4
rd_DC = np.array([[0.1, 0.02, 0]]).T # vecteur de C à D (d1, d2, d3)
D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 5
re_ED = np.array([[0.3, 0, 0]]).T # vecteur de D à E ( e1, e2, e3)
E = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 6
rt_TE = np.array([[0.02, 0, 0]]).T # vecteur de E à T (t1, t2, t3)
T = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere outil
# piece
rw_PW = np.array([[0.5994, 0.1991, 0]]).T # position de la piece dans le repere monde
Po = np.array([[0, 0, 0]]).T # position de la piece dans le repere monde
P = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere piece
rp_LbP = np.array([[0.15, 0, 0]]).T # longueur de la piece
rp_HgP = np.array([[0, 0.1, 0]]).T # grande hauteur de la piece
rp_HdP = np.array([[0, 0.05, 0]]).T # petite hauteur de la piece
# caméra
rw_VW = np.array([[0.8, 0.7, 0]]).T # position de la caméra dans le repere monde
V = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere camera
# point de la trance (repere camera)
TI = np.array([[0.158920, 0.157470, 0.153781, 0.152420, 0.150931],
             [0.013914, 0.021067, 0.039266, 0.045970, 0.053326],
             [0.028686, 0.008891, -0.040587, -0.060395, -0.080185]])
# défaut de la piece (repere caméra)
DF = np.array([[0.153758, 0.145698, 0.153932, 0.152097, 0.146104],
               [0.039379, 0.079138, 0.038521, 0.047573, 0.077134],
               [-0.025575, -0.039398, 0.009411, 0.035692, 0.030571]])
## début du code
rw_PW = mat_rot_ang(PI/2, 1) @ rw_PW
rp_LbP = mat_rot_ang(PI/2, 1) @ rp_LbP
rp_HgP = mat_rot_ang(PI/2, 1) @ rp_HgP
rp_HdP = mat_rot_ang(PI/2, 1) @ rp_HdP

wRa = mat_rot_ang(qT[0], 2)
aRb = mat_rot_ang(qT[1], 3)
bRc = mat_rot_ang(qT[2], 3)
cRd = mat_rot_ang(qT[3], 1)
dRe = mat_rot_ang(qT[4], 3)
eRt = mat_rot_ang(qT[5], 1)

rw_AW = wRa @ ra_AWo
rw_BW = wRa @ aRb @ rb_BA
rw_CW = wRa @ aRb @ bRc @ rc_CB
rw_DW = wRa @ aRb @ bRc @ cRd @ rd_DC
rw_EW = wRa @ aRb @ bRc @ cRd @ dRe @ re_ED
rw_TW = wRa @ aRb @ bRc @ cRd @ dRe @ eRt @ rt_TE

# calcul des défaut selon le repere piece
rv_DF_1 = np.array ([DF[:, 0]]).T 
rw_DF1W = (-1*rv_DF_1)  +  (-1* rw_VW)
rt_DF1T = rw_DF1W + rw_TW
rp_DF1P = (-1*rt_DF1T) + (rw_PW - rw_TW) 

rv_DF_2 = np.array([DF[:, 1]]).T 
rw_DF2W = (-1*rv_DF_2)  +  (-1* rw_VW)
rt_DF2T = rw_DF2W + rw_TW
rp_DF2P = (-1*rt_DF2T) + (rw_PW - rw_TW) 

rv_DF_3 = np.array([DF[:, 2]]).T 
rw_DF3W = (-1*rv_DF_3)  +  (-1* rw_VW)
rt_DF3T = rw_DF3W + rw_TW
rp_DF3P = (-1*rt_DF3T) + (rw_PW - rw_TW)

rv_DF_4 = np.array([DF[:, 3]]).T 
rw_DF4W = (-1*rv_DF_4)  +  (-1* rw_VW)
rt_DF4T = rw_DF4W + rw_TW
rp_DF4P = (-1*rt_DF4T) + (rw_PW - rw_TW)

rv_DF_5 = np.array([DF[:, 4]]).T 
rw_DF5W = (-1*rv_DF_5)  +  (-1* rw_VW)
rt_DF5T = rw_DF5W + rw_TW
rp_DF5P = (-1*rt_DF5T) + (rw_PW - rw_TW)


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
rw_HdW = rw_LbW + rp_HdP
Piece_W = np.array([rw_PW, rw_LbW, rw_HdW, rw_HgW, rw_PW])
Piece_w1 = np.array([Piece_W[:,0]])
Piece_w3 = np.array([Piece_W[:,2]])
Piece_w2 = np.array([Piece_W[:,1]])
#dessiner les membres du robot
pt_A = rw_AW
pt_B = pt_A + rw_BW
pt_C = pt_B + rw_CW
pt_D = pt_C + rw_DW
pt_E = pt_D + rw_EW
pt_T = pt_E + rw_TW
Robot = np.array([Wo, pt_A, pt_B, pt_C, pt_D, pt_E, pt_T])
Robot_w1 = np.array([Robot[:,0]])
Robot_w3 = np.array([Robot[:,2]])
Robot_w2 = np.array([Robot[:,1]])
print("Tool position : ", pt_T)  


# affichage de la configuration du robot
ax.plot(Robot_w1, Robot_w3, Robot_w2, 'r.-', label = "Robot") # robot
# afficher la pièce à inspecter
ax.plot(Piece_w1, Piece_w3, Piece_w2, 'k.-', label = "Piece") # pièce à inspecter
#afficher les# points de référence
ax.plot([Wo[0]],[Wo[2]], [Wo[1]], 'bo-', label = "World") # base du robot
ax.plot([pt_T[0]],[pt_T[2]], [pt_T[1]], 'co-', label = "Tool") # position de l'outil
ax.plot([rw_VW[0]],[rw_VW[2]], [rw_VW[1]], 'mo-', label = "Camera") # position de la caméra
ax.plot([rw_PW[0]],[rw_PW[2]], [rw_PW[1]], 'go-', label = "Piece") # position de la pièce
# plt.legend()
plt.show()

# afficher les défauts
defaut = plt.figure()
ax2 = defaut.add_subplot(111, projection='3d')
rp_HdP = rp_LbP + rp_HdP
Piece_P = np.array([Po, rp_LbP, rp_HdP, rp_HgP, Po])
Piece_p1 = np.array([Piece_P[:,0]])
Piece_p3 = np.array([Piece_P[:,2]])
Piece_p2 = np.array([Piece_P[:,1]])
ax2.plot([rp_DF1P[0]],[rp_DF1P[1]], [rp_DF1P[2]], 'ro-', label = "DF1") # défaut 1
ax2.plot([rp_DF2P[0]],[rp_DF2P[1]], [rp_DF2P[2]], 'go-', label = "DF2") # défaut 2
ax2.plot([rp_DF3P[0]],[rp_DF3P[1]], [rp_DF3P[2]], 'bo-', label = "DF3") # défaut 3
ax2.plot([rp_DF4P[0]],[rp_DF4P[1]], [rp_DF4P[2]], 'co-', label = "DF4") # défaut 4
ax2.plot([rp_DF5P[0]],[rp_DF5P[1]], [rp_DF5P[2]], 'mo-', label = "DF5") # défaut 5
ax2.plot(Piece_p1, Piece_p3, Piece_p2, 'k.-', label = "Piece") # pièce à inspecter



plt.show()



