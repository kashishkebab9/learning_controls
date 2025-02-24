import numpy as np
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed

# Pendulum parameters
g = 9.81  
m = .1  
L = 1.0  

# State and control discretization
X1_RANGE_MIN, X1_RANGE_MAX = 0, 2*np.pi
X2_RANGE_MIN, X2_RANGE_MAX = -10, 10
U_RANGE_MIN, U_RANGE_MAX = -3, 3

X1_NUM_BINS, X2_NUM_BINS, U_NUM_BINS = 151, 151, 21
X1_BINS = np.linspace(X1_RANGE_MIN, X1_RANGE_MAX, num=X1_NUM_BINS)
X2_BINS = np.linspace(X2_RANGE_MIN, X2_RANGE_MAX, num=X2_NUM_BINS)
U_BINS = np.linspace(U_RANGE_MIN, U_RANGE_MAX, num=U_NUM_BINS)

def get_closest_grid_idx(x1, x2):
    return np.array([np.argmin(np.abs(X1_BINS - x1)), np.argmin(np.abs(X2_BINS - x2))])

def dynamics_next_state(x1, x2, u, dt=0.1):
    x1_next = x1 + x2 * dt
    angular_acc = dt * ((u - (m*g*L*np.sin(x1))) / (m*L**2))
    x2_next = angular_acc + x2
    return np.array([x1_next, x2_next])

def calculate_loss(x1, x2, u):
    #return np.linalg.norm([x1 - np.pi, x2])  
    return 2*(x1-np.pi)**2 + 2 * x2**2 + u**2

def update_state_value(x1_idx, x2_idx, Z, gamma):
    x1_val = X1_BINS[x1_idx]
    x2_val = X2_BINS[x2_idx]
    min_loss = sys.float_info.max
    u_opt = 0

    for u in U_BINS:
        x_next = dynamics_next_state(x1_val, x2_val, u)
        x_next_idx = get_closest_grid_idx(x_next[0], x_next[1])
        
        if 0 <= x_next_idx[0] < X1_NUM_BINS and 0 <= x_next_idx[1] < X2_NUM_BINS:
            cost = calculate_loss(x1_val, x2_val, u) + gamma * Z[x_next_idx[0], x_next_idx[1]]
            if cost < min_loss:
                min_loss = cost
                u_opt = u

    return x1_idx, x2_idx, min_loss, u_opt

if __name__ == '__main__':
    X1, X2 = np.meshgrid(X1_BINS, X2_BINS)
    Z = np.zeros_like(X1)
    U_OPT = np.zeros_like(Z)
    goal = np.array([np.pi, 0])
    goal_idx = get_closest_grid_idx(goal[0], goal[1])
    Z[goal_idx[0], goal_idx[1]] = 0

    max_iterations = 500
    tolerance = 0.01
    gamma = 0.95

    num_cores = -1  # Use all available cores

    for iter in range(max_iterations):
        Z[goal_idx[0], goal_idx[1]] = 0
        Z_prev = Z.copy()

        # Parallel computation
        results = Parallel(n_jobs=num_cores)(
            delayed(update_state_value)(x1_idx, x2_idx, Z_prev, gamma)
            for x1_idx in range(X1_NUM_BINS)
            for x2_idx in range(X2_NUM_BINS)
        )

        # Update Z and U_OPT based on results
        for x1_idx, x2_idx, min_loss, u_opt in results:
            Z[x1_idx, x2_idx] = min_loss
            U_OPT[x1_idx, x2_idx] = u_opt

        max_diff = np.max(np.abs(Z - Z_prev))
        print(f"Iteration {iter+1}, Max Difference: {max_diff}")

        if max_diff < tolerance:
            print("Converged")
            break

    # Plot results
    plt.figure(1, figsize=(9, 4))
    plt.imshow(Z, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='Value Function (Z)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Value Function Heatmap')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(U_OPT, extent=[X1_RANGE_MIN, X1_RANGE_MAX, X2_RANGE_MIN, X2_RANGE_MAX], origin='lower', cmap='viridis')
    plt.colorbar(label='Optimal U')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Optimal Control')
    plt.show()

    # Trajectory rollout
    dt = 0.1
    time_horizon = 30
    steps = int(time_horizon / dt)
    x1, x2 = 0, 0.2
    trajectory = []
    controls = []

    for _ in range(steps):
        x1_idx, x2_idx = get_closest_grid_idx(x1, x2)
        u = U_OPT[x1_idx, x2_idx]
        trajectory.append([x1, x2])
        controls.append(u)
        x1, x2 = dynamics_next_state(x1, x2, u, dt)

    trajectory = np.array(trajectory)
    times = np.linspace(0, time_horizon, steps)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(times, trajectory[:, 0], label='Angle (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Angle vs Time')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(times, controls, label='Control Input')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Torque (Nm)')
    plt.title('Control Input vs Time')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Phase Plot')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angle vs Angular Velocity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

