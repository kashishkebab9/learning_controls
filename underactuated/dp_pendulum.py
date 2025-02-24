import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys

# pendulum parameters
g = 9.81  # acceleration due to gravity (m/s^2)
m = 1.0  # mass of the pendulum (kg)
L = 1.0  # length of the pendulum (m)

# min/max params for states and controls
X1_RANGE_MIN = 0
X1_RANGE_MAX = 2*np.pi
X2_RANGE_MIN = -10
X2_RANGE_MAX = 10
U_RANGE_MIN = -3
U_RANGE_MAX = 3

X1_NUM_BINS = 101
X2_NUM_BINS = 101
U_NUM_BINS = 9
X1_BINS = np.linspace(X1_RANGE_MIN, X1_RANGE_MAX, num=X1_NUM_BINS)
X2_BINS = np.linspace(X2_RANGE_MIN, X2_RANGE_MAX, num=X2_NUM_BINS)
U_BINS = np.linspace(U_RANGE_MIN, U_RANGE_MAX, num=U_NUM_BINS)

def get_closest_grid_idx(x1, x2):
    # return the idx of the continous value (meant mostly for goal)
    return np.array([np.argmin(np.abs(X1_BINS - x1)), np.argmin(np.abs(X2_BINS - x2))])

    
def dynamics_next_state(x1, x2, u, dt=.1):
    # Takes in current state and u and returns what x_n+1 is
    x1_next = x1 + x2 * dt
    angular_acc = dt * ((u - (m*g*L*np.sin(x1)))/ (m*L**2) )
    x2_next = angular_acc + x2
    return np.array([x1_next, x2_next])

def calculate_loss(x1, x2, u):
    # Takes in current state and u and returns what the cost is for that current state
    return np.linalg.norm([x1-np.pi, x2]) 

def find_neighbors(x1, x2):
    # finds the neighbors, up to the limits
    idx = get_closest_grid_idx(x1, x2)

    n_up = None
    n_right = None
    n_left = None
    n_down = None

    n_right = np.array([idx[0]+1, idx[1]])
    n_left = np.array([idx[0]-1, idx[1]])
    n_up = np.array([idx[0], idx[1]+1])
    n_down = np.array([idx[0], idx[1]-1])

    neighbors = []

    if idx[0] != 0:
        neighbors.append(n_left)
    if idx[0] != X1_NUM_BINS -1:
        neighbors.append(n_right)
    if idx[1] != 0:
        neighbors.append(n_down)
    if idx[1] != X2_NUM_BINS -1:
        neighbors.append(n_up)

    return neighbors

if __name__ == '__main__':
    X1, X2 = np.meshgrid(X1_BINS, X2_BINS)
    Z = np.zeros_like(X1)
    U_OPT = np.zeros_like(Z)
    goal = np.array([np.pi,0])
    goal_idx = get_closest_grid_idx(goal[0], goal[1])
    Z[goal_idx[0], goal_idx[1]] = 0

    max_iterations = 100
    tolerance = .1
    gamma = .95

    for iter in range(max_iterations):
        Z[goal_idx[0], goal_idx[1]] = 0
        Z_prev = Z.copy()
        for x1_idx in range(X1_NUM_BINS):
            for x2_idx in range(X2_NUM_BINS):
                x1_val = X1_BINS[x1_idx]
                x2_val = X2_BINS[x2_idx]
                min_loss = sys.float_info.max 
                u_opt = 0

                for u in U_BINS:
                    x_next = dynamics_next_state(x1_val, x2_val, u)
                    x_next_idx = get_closest_grid_idx(x_next[0], x_next[1])
                    
                    if 0 <= x_next_idx[0] < X1_NUM_BINS and 0 <= x_next_idx[1] < X2_NUM_BINS:
                        cost = calculate_loss(x1_val, x2_val, u) + gamma* Z[x_next_idx[0], x_next_idx[1]]
                        if cost < min_loss:
                            min_loss = cost
                            u_opt = u

                Z[x1_idx, x2_idx] = min_loss
                U_OPT[x1_idx, x2_idx] = u_opt

        if np.max(np.abs(Z - Z_prev)) < tolerance:
            print("reached convergence")
            break
        else:
            print(np.max(np.abs(Z - Z_prev)))


    print(Z)

    plt.figure(1,figsize=(9, 4))
    plt.imshow(Z, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='Value Function (Z)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Value Function Heatmap')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(U_OPT, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='OPTIMAL U')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Optimal Control')
    plt.show()
