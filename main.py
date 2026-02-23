# APP5
# Modélisation cinématique d'un systeme robotisé 6 axes d'inspection par vision


import numpy as np
import matplotlib.pyplot as plt

World = np.array([0, 0, 0])# origine du repere monde
W = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere monde
Tool = np.array([1, 0, 0]) # position de l'outil dans le repere monde
T = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]) # base du repere outil
Piece = np.array([0, 1, 0]) # position de la piece dans le repere monde
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






# affichage de la configuration du robot
A = World + Ao
B = A + Bo
C = B + Co
D = C + Do
E = D + Eo
T = E + To

plt.plot([E[0], T[0]], [E[1], T[1]], 'k.-', label = "joint 6") # joint 6
plt.plot([D[0], E[0]], [D[1], E[1]], 'y.-', label = "joint 5") # joint 5
plt.plot([C[0], D[0]], [C[1], D[1]], 'm.-', label = "joint 4") # joint 4
plt.plot([B[0], C[0]], [B[1], C[1]], 'c.-', label = "joint 3") # joint 3
plt.plot([A[0], B[0]], [A[1], B[1]], 'b.-', label = "joint 2") # joint 2
plt.plot([World[0],A[0]], [World[1], A[1]], 'g.-', label = "joint 1") # joint 1
plt.plot([World[0]],[World[1]], 'ro-', label = "World") # base du robot
plt.plot([T[0]],[T[1]], 'ro-', label = "Tool") # position de l'outil
plt.plot([Vo[0]],[Vo[1]], 'ko-', label = "Camera") # position de la caméra
plt.legend()
plt.show()






