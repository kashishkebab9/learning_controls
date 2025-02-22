import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys

# min/max params for states and controls
X1_RANGE_MIN = -50
X1_RANGE_MAX = 50
X2_RANGE_MIN = -50
X2_RANGE_MAX = 50
U_RANGE_MIN = -20
U_RANGE_MAX = 20

X1_NUM_BINS = 301
X2_NUM_BINS = 301
U_NUM_BINS = 101
X1_BINS = np.linspace(X1_RANGE_MIN, X1_RANGE_MAX, num=X1_NUM_BINS)
X2_BINS = np.linspace(X2_RANGE_MIN, X2_RANGE_MAX, num=X2_NUM_BINS)
U_BINS = np.linspace(U_RANGE_MIN, U_RANGE_MAX, num=U_NUM_BINS)

print(X1_BINS)

def get_closest_grid_idx(x1, x2):
    # return the idx of the continous value (meant mostly for goal)
    return np.array([np.argmin(np.abs(X1_BINS - x1)), np.argmin(np.abs(X2_BINS - x2))])

def dynamics_next_state(x1, x2, u):
    # Takes in current state and u and returns what x_n+1 is
    x1_next = x1 + x2
    x2_next = x2 + u
    return np.array([x1_next, x2_next])

def calculate_loss(x1, x2, u):
    # Takes in current state and u and returns what the cost is for that current state
    return u**2

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
    Z.fill(100)
    goal = np.array([0,0])
    goal_idx = get_closest_grid_idx(goal[0], goal[1])
    Z[goal_idx[0], goal_idx[1]] = 0


    # both completed indices and queue_to_complete hold indices!
    completed_indices = set()
    completed_indices.add(tuple(goal_idx))

    queue_to_complete = deque()
    goal_neighbors = find_neighbors(goal[0], goal[1])
    for i in goal_neighbors:
        queue_to_complete.append(tuple(i))

    while len(queue_to_complete) != 0:
        # get first item from queue
        Z_prev = Z.copy()
        element = queue_to_complete[0]
        x1_val = X1_BINS[element[0]]
        x2_val = X2_BINS[element[1]]

        # calculate the loss function for all u in U_BINS
        min_loss = sys.float_info.max 
        for u in U_BINS:
            x_next = dynamics_next_state(x1_val, x2_val, u)
            x_next_idx = get_closest_grid_idx(x_next[0], x_next[1])
            
            if 0 <= x_next_idx[0] < X1_NUM_BINS and 0 <= x_next_idx[1] < X2_NUM_BINS:
                cost = calculate_loss(x1_val, x2_val, u) + Z[x_next_idx[0], x_next_idx[1]]
                min_loss = min(min_loss, cost)


        Z[element[0], element[1]] = min_loss

        # pop at the end
        queue_to_complete.popleft()
        completed_indices.add(tuple(element))
        neighbors = find_neighbors(x1_val, x2_val)

        for i in neighbors:
            if tuple(i) not in completed_indices and tuple(i) not in queue_to_complete:
                queue_to_complete.append(tuple(i))

        
    
    print("done")
    print(Z)

    plt.figure(figsize=(10, 8))
    plt.imshow(Z, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='Value Function (Z)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Value Function Heatmap')
    plt.show()





