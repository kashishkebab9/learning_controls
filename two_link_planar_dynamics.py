import math
import numpy as np

l_1 =
l_2 =
theta_1 =
theta_2


joint_1_kin = np.matrix([[l_1*math.cos(theta_1)], [l_1*math.sin(theta_1)]])
joint_1_dot = np.matrix([[-l_1*math.sin(theta_1)], [l_1*math.cos(theta_1)]])

joint_2_kin = np.matrix([[l_1 * math.cos(theta_1) + l_2 * math.cos(theta_1 + theta_2)], [l_1*math.sin(theta_1) + l_2*math.sin(theta_1 + theta_2)]]) 
joint_2_dot = np.matrix([[-l_1*math.sin(theta_1)], [l_1*math.cos(theta_1)]]