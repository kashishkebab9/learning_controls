clear all;
clc;

%parameters
g = 9.81
L = 1
M = .25

A = [0, 1; g/L 0];
B = [0; 1/(M*L)];
C = [1, 0];
D = [0];

sys = ss(A, B, C, D)


P = [-1, -3.1321]
K = place(A,B,P)
E = eig(A)
Acl = A - B*K;
Ecl = eig(Acl)

syscl =ss(Acl, B, C, D)


Kdc = dcgain(syscl)
Kr = 1/Kdc

syscl_scaled = ss(Acl, B*Kr, C, D);
step(syscl_scaled)


