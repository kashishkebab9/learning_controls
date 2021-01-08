from control.matlab import *
import math
import numpy as np
g = 9.81
L = 1
M = .45
A = np.matrix([[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1], [0, 0, g/L, 0]])
B = np.matrix([[0],[1/M],[0], [1/(M*L)]])
C = np.matrix([[1, 0, 0, 0]])
D = np.matrix([[0]])
 
sys = ss(A, B, C, D)

E = np.linalg.eigvals(A)
print (E)