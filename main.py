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
def changement_base(vecteur, R):
    """ 
    Effectue un changement de base du vecteur de la base1 à la base2 
    vecteur : vecteur à transformer
    R : matrice de rotation pour passer de la base1 à la base2
    """
    new_vector =  np.dot(R, vecteur)
    return new_vector

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

# affichage de la simulation
def afficher_robot(Po, rAWo, rBA, rCB, rDC, rED, rTE, Wo, Lb, Hg, Hd):
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
    Dlb = Po + Lb
    Dhg = Po + Hg
    Dhd = Dlb + Hd
    #dessiner les membres du robot
    a = Wo + rAWo
    b = a + rBA
    c = b + rCB
    d = c + rDC
    e = d + rED
    t = e + rTE
    print("Tool position : ", t)  
    # affichage de la configuration du robot
    ax.plot([e[0], t[0]], [e[2], t[2]], [e[1], t[1]], 'k.-', label = "joint 6") # joint 6
    ax.plot([d[0], e[0]], [d[2], e[2]], [d[1], e[1]], 'y.-', label = "joint 5") # joint 5
    ax.plot([c[0], d[0]], [c[2], d[2]], [c[1], d[1]], 'm.-', label = "joint 4") # joint 4
    ax.plot([b[0], c[0]], [b[2], c[2]], [b[1], c[1]], 'c.-', label = "joint 3") # joint 3
    ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], 'b.-', label = "joint 2") # joint 2
    ax.plot([Wo[0],a[0]], [Wo[2], a[2]], [Wo[1], a[1]], 'g.-', label = "joint 1") # joint 1
    # afficher la pièce à inspecter
    ax.plot([Po[0], Dlb[0], Dhd[0], Dhg[0], Po[0]], [Po[2], Dlb[2], Dhd[2], Dhg[2], Po[2]],
            [Po[1], Dlb[1], Dhd[1], Dhg[1], Po[1]], 'k.-', label = "Piece") # pièce à inspecter
    #afficher les# points de référence
    ax.plot([Wo[0]],[Wo[2]], [Wo[1]], 'ro-', label = "World") # base du robot
    ax.plot([t[0]],[t[2]], [t[1]], 'ro-', label = "Tool") # position de l'outil
    ax.plot([Vo[0]],[Vo[2]], [Vo[1]], 'ko-', label = "Camera") # position de la caméra
    ax.plot([Po[0]],[Po[2]], [Po[1]], 'go-', label = "Piece") # position de la pièce
    # plt.legend()
    plt.show()


## initialisation des données
PI = float(np.pi)  # 3.141592653
# pour tout les matrice, premiere rangée: x(1), deuxieme rangée: y(2), troisieme rangée: z(3)
#qT = np.array([0, 0, 0, 0, 0, 0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta
qT = np.array([0, -0.3, 0, 0, 0.5, -1.6]) # inspection de la face avant de la pièce
#qT = np.array([-0.4, -1.2, 0, 0, -0.3708, 0]) # prise de la pièce
#qT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # configuration de validation

Wo = np.array([0, 0, 0])# origine du repere world
W = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere world
ra_AWo = np.array([0, 0.15, 0]) # vecteur de W à A (a1, a2, a3)
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 2
rb_BA = np.array([0.05, 0.1, 0]) # vecteur de A à B (b1, b2, b3)
B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 3
rc_CB = np.array([0, 0.5, 0]) # vecteur de B à C (c1, c2, c3)
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 4
rd_DC = np.array([0.1, 0.02, 0]) # vecteur de C à D (d1, d2, d3)
D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 5
re_ED = np.array([0.3, 0, 0]) # vecteur de D à E ( e1, e2, e3)
E = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 6
rt_TE = np.array([0.02, 0, 0]) # vecteur de E à T (t1, t2, t3)
T = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere outil
# piece
Po = np.array([0.5994, 0.1991, 0]) # position de la piece dans le repere monde
P = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere piece
Lb = np.array([0.15, 0, 0]) # longueur de la piece
Hg = np.array([0, 0.1, 0]) # grande hauteur de la piece
Hd = np.array([0, 0.05, 0]) # petite hauteur de la piece
# caméra
Vo = np.array([0.8, 0.7, 0]) # position de la caméra dans le repere monde
V = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere camera

## début du code
Po = np.dot(mat_rot_ang(PI/2, 1), Po)
Lb = np.dot(mat_rot_ang(PI/2, 1), Lb)
Hg = np.dot(mat_rot_ang(PI/2, 1), Hg)
Hd = np.dot(mat_rot_ang(PI/2, 1), Hd)

wRa = mat_rot_ang(qT[0], 2)
aRb = mat_rot_ang(qT[1], 3)
bRc = mat_rot_ang(qT[2], 3)
cRd = mat_rot_ang(qT[3], 1)
dRe = mat_rot_ang(qT[4], 3)
eRt = mat_rot_ang(qT[5], 1)

rw_AW = changement_base(ra_AWo, wRa)
rw_BW = changement_base(rb_BA, wRa@aRb)
rw_CW = changement_base(rc_CB, wRa@aRb@bRc)
rw_DW = changement_base(rd_DC, wRa@aRb@bRc@cRd)
rw_EW = changement_base(re_ED, wRa@aRb@bRc@cRd@dRe)
rw_TW = changement_base(rt_TE, wRa@aRb@bRc@cRd@dRe@eRt)

afficher_robot(Po, rw_AW, rw_BW, rw_CW, rw_DW, rw_EW, rw_TW, Wo, Lb, Hg, Hd)






