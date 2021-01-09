from control.matlab import *
import math
import numpy as np
g = 9.81
L = 1
M = .5
A = np.matrix([[0, 1], [g/L, 0]])
B = np.matrix([[0], [1/(M*L)]])

P = np.matrix([[ -1, -3.1321]])
E = np.linalg.eigvals(A)
print (E)

K = place(A, B, P)