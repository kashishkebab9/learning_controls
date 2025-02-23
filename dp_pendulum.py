import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys

# pendulum parameters
g = 9.81  # acceleration due to gravity (m/s^2)
m = 1.0  # mass of the pendulum (kg)
L = 1.0  # length of the pendulum (m)

# min/max params for states and controls
X1_RANGE_MIN = -2*np.pi
X1_RANGE_MAX = 2*np.pi
X2_RANGE_MIN = -5
X2_RANGE_MAX = 5
U_RANGE_MIN = -1
U_RANGE_MAX = 1

X1_NUM_BINS = 501
X2_NUM_BINS = 501
U_NUM_BINS = 11
X1_BINS = np.linspace(X1_RANGE_MIN, X1_RANGE_MAX, num=X1_NUM_BINS)
X2_BINS = np.linspace(X2_RANGE_MIN, X2_RANGE_MAX, num=X2_NUM_BINS)
# U_BINS = np.linspace(U_RANGE_MIN, U_RANGE_MAX, num=U_NUM_BINS)
U_BINS = [U_RANGE_MIN, U_RANGE_MAX]

def get_closest_grid_idx(x1, x2):
    # return the idx of the continous value (meant mostly for goal)
    return np.array([np.argmin(np.abs(X1_BINS - x1)), np.argmin(np.abs(X2_BINS - x2))])

def dynamics_next_state(x1, x2, u, dt=.1):
    # Takes in current state and u and returns what x_n+1 is
    x1_next = x1 + x2 * dt
    angular_acc = dt * ((u - (m*g*L*np.sin(x1)))/ (m*L**2) )
    x2_next = angular_acc + x2
    return np.array([x1_next, x2_next])

def calculate_loss(x1, x2, u, alpha=1, beta=.5):
    # Loss function: penalizing both position and control effort
    return x1**2 + x2**2 + u**2

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

    goal = np.array([np.pi,.1])
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
        element = queue_to_complete[0]
        x1_val = X1_BINS[element[0]]
        x2_val = X2_BINS[element[1]]

        # calculate the loss function for all u in U_BINS
        optimal_u = 0
        min_loss = sys.float_info.max 
        for u in U_BINS:
            x_next = dynamics_next_state(x1_val, x2_val, u)
            x_next_idx = get_closest_grid_idx(x_next[0], x_next[1])
            
            if 0 <= x_next_idx[0] < X1_NUM_BINS and 0 <= x_next_idx[1] < X2_NUM_BINS:
                cost = calculate_loss(x1_val, x2_val, u) + Z[x_next_idx[0], x_next_idx[1]]
                if cost < min_loss:
                    min_loss = cost
                    optimal_u = u



        U_OPT[element[0], element[1]] = optimal_u
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
    plt.colorbar(label='Cost Function')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Value Function Heatmap')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(U_OPT, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='OPTIMAL U')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Value Function Heatmap')
    plt.show()

    ## Simulation
    # Choose an initial state (start point)
    x1_start = 0  # initial angle (in degrees or radians)
    x2_start = 0   # initial angular velocity (in radians/s)

    # Define the time step and simulation duration
    dt = 0.1
    time_steps = 200  # number of time steps

    # Initialize lists to store the trajectory and control inputs
    x1_trajectory = [x1_start]
    x2_trajectory = [x2_start]
    u_trajectory = []

    # Current state
    x1_current, x2_current = x1_start, x2_start

    # Simulate the trajectory
    for _ in range(time_steps):
        # Get the index of the current state
        idx = get_closest_grid_idx(x1_current, x2_current)
        
        # Get the optimal control for the current state
        u_opt = U_OPT[idx[0], idx[1]]
        u_trajectory.append(u_opt)
        
        # Update the state using the dynamics and the optimal control
        x_next = dynamics_next_state(x1_current, x2_current, u_opt, dt)
        x1_current, x2_current = x_next[0], x_next[1]
        
        # Store the new state
        x1_trajectory.append(x1_current)
        x2_trajectory.append(x2_current)

    # Convert the lists to numpy arrays for easier plotting
    x1_trajectory = np.array(x1_trajectory)
    x2_trajectory = np.array(x2_trajectory)

    # Plot the state trajectory in the (x1, x2) space
    plt.figure(figsize=(10, 6))
    plt.plot(x1_trajectory, x2_trajectory, label='Trajectory', color='g')
    plt.xlabel('Angle (x1)')
    plt.ylabel('Angular Velocity (x2)')
    plt.title('State Space Trajectory')
    plt.grid(True)
    plt.show()

    # Plot the control input trajectory over time
    plt.figure(figsize=(10, 6))
    plt.plot(x1_trajectory, label='Optimal Control (u)', color='b')
    plt.xlabel('Time Step')
    plt.ylabel('angle')
    plt.title('Optimal Control Input Over Time')
    plt.grid(True)
    plt.show()
