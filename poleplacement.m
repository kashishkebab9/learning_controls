clc;
clear all;

g = 9.81
L = 1
M = .45
A = [0, 1, 0, 0;0, 0, 0, 0; 0, 0, 0, 1; 0, 0, g/L, 0]
B = [0; 1/M; 0; 1/(M*L)]
C = [1, 0, 0, 0]
D = [0]
F = [ -3, 0, 0, 0]
sys = ss(A, B, C, D)

x =rank(B)
E = eig(A)


K = place(A, B, F)