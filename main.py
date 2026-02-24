# APP5
# Modélisation cinématique d'un systeme robotisé 6 axes d'inspection par vision


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

# initialisation des données
# pour tout les matrice, premiere range: x, deuxieme range: y, troisieme range: z
Wo = np.array([0, 0, 0])# origine du repere monde
W = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere monde
T = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere outil
Po = np.array([0.5994, 0, 0.1991]) # position de la piece dans le repere monde
P = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere piece
Vo = np.array([0.8, 0.7, 0]) # position de la caméra dans le repere monde
V = np.array([[1, 0, 0],
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere camera
qT = np.array([0,0,0,0,0,0]) # qT = [q1, q2, q3, q4, q5, q6] où q = theta
Ao = np.array([0, 0.15, 0]) # vecteur de W à A (a1, a2, a3)
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 2
Bo = np.array([0.05, 0.1, 0]) # vecteur de A à B (b1, b2, b3)
B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 3
Co = np.array([0, 0.5, 0]) # vecteur de B à C (c1, c2, c3)
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 4
Do = np.array([0.1, 0.02, 0]) # vecteur de C à D (d1, d2, d3)
D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 5
Eo = np.array([0.3, 0, 0]) # vecteur de D à E ( e1, e2, e3)
E = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) # base du joint 6
To = np.array([0.02, 0, 0]) # vecteur de E à T (t1, t2, t3)
Lb = np.array([0.15, 0, 0]) # longueur de la piece
Hg = np.array([0, 0, 0.1]) # grande hauteur de la piece
Hd = np.array([0, 0, 0.05]) # petite hauteur de la piece

# rotation des bases de chaque joint
def rotation(base, theta, axe):
    """ 
    Effectue une rotation de la base autour de l'axe spécifié par l'angle theta (en radians) 
    base : base de reference
    theta : angle de rotation en radians
    axe : axe de rotation (1 pour x, 2 pour y, 3 pour z) 
    """
    new_base = np.zeros((3, 3))
    if axe == 1:
        new_base[0,0] = 1
        new_base[1,1] = np.cos(theta)
        new_base[1,2] = -np.sin(theta)
        new_base[2,1] = np.sin(theta)
        new_base[2,2] = np.cos(theta)
    elif axe == 2:
        new_base[0,0] = np.cos(theta)
        new_base[0,2] = np.sin(theta)
        new_base[1,1] = 1
        new_base[2,0] = -np.sin(theta)
        new_base[2,2] = np.cos(theta)
    elif axe == 3:
        new_base[0,0] = np.cos(theta)
        new_base[0,1] = -np.sin(theta)
        new_base[1,0] = np.sin(theta)
        new_base[1,1] = np.cos(theta)
        new_base[2,2] = 1
    return np.dot(new_base, base)
# matrice de rotation à partir de deux bases
def matrice_rotation(base1, base2):
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
def changement_base(vecteur, base1, base2):
    """ 
    Effectue un changement de base du vecteur de la base1 à la base2 
    vecteur : vecteur à transformer
    base1 : base de référence actuelle du vecteur
    base2 : base de référence souhaitée du vecteur
    """
    new_vecteur = np.zeros(3)
    vect_unitaire = vecteur / np.linalg.norm(vecteur) # normalisation du vecteur
    new_vector =  np.dot(matrice_rotation(base1, base2), vecteur)
    return new_vector

# affichage de la simulation
def afficher_robot(Po, Ao, Bo, Co, Do, Eo, To, Wo, Lb, Hg, Hd):
    # setup de l'affichage
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.set_xlabel('W1')
    ax.set_ylabel('W3')
    ax.set_zlabel('W2')

    # dessiner la piece
    Dlb = Po + Lb
    Dhg = Po + Hg
    Dhd = Dlb + Hd
    #dessiner les membres du robot
    A = Wo + Ao
    B = A + Bo
    C = B + Co
    D = C + Do
    E = D + Eo
    T = E + To
    # affichage de la configuration du robot
    ax.plot([E[0], T[0]], [E[2], T[2]], [E[1], T[1]], 'k.-', label = "joint 6") # joint 6
    ax.plot([D[0], E[0]], [D[2], E[2]], [D[1], E[1]], 'y.-', label = "joint 5") # joint 5
    ax.plot([C[0], D[0]], [C[2], D[2]], [C[1], D[1]], 'm.-', label = "joint 4") # joint 4
    ax.plot([B[0], C[0]], [B[2], C[2]], [B[1], C[1]], 'c.-', label = "joint 3") # joint 3
    ax.plot([A[0], B[0]], [A[2], B[2]], [A[1], B[1]], 'b.-', label = "joint 2") # joint 2
    ax.plot([Wo[0],A[0]], [Wo[2], A[2]], [Wo[1], A[1]], 'g.-', label = "joint 1") # joint 1
    # afficher la pièce à inspecter
    ax.plot([Po[0], Dlb[0], Dhd[0], Dhg[0], Po[0]], [Po[2], Dlb[2], Dhd[2], Dhg[2], Po[2]],
            [Po[1], Dlb[1], Dhd[1], Dhg[1], Po[1]], 'k.-', label = "Piece") # pièce à inspecter
    #afficher les# points de référence
    ax.plot([Wo[0]],[Wo[2]], [Wo[1]], 'ro-', label = "World") # base du robot
    ax.plot([T[0]],[T[2]], [T[1]], 'ro-', label = "Tool") # position de l'outil
    ax.plot([Vo[0]],[Vo[2]], [Vo[1]], 'ko-', label = "Camera") # position de la caméra
    ax.plot([Po[0]],[Po[2]], [Po[1]], 'go-', label = "Piece") # position de la pièce
    # plt.legend()
    plt.show()


afficher_robot(Po, Ao, Bo, Co, Do, Eo, To, Wo, Lb, Hg, Hd)





