from whisker_driver import WhiskerArraySimEnv
from whisker_driver import read_object_path_xy
from whisker_driver import calc_path_data
from whisker_driver import distance_to_curve_2d
import random
import numpy as np
import matplotlib.pyplot as plt

def calc_step_reward(loc_steps, path_data, wake_width=100, L0=10, RL_max_step=99):
    """
    Reward for every RL action

    path_data - array, n*3, [x,y,s] 
    wake_width - effective range to sense the wake
    """
    rewards = []
    vertical_distances = []
    progres_distances = []

    for step in range(1, RL_max_step):
        old_s = loc_steps[step-1,:]
        new_s = loc_steps[step,:]

        d_old, i_old = distance_to_curve_2d(old_s[0], old_s[1], path_data[:,0:2])
        d_new, i_new = distance_to_curve_2d(new_s[0], new_s[1], path_data[:,0:2])
        l_old = path_data[i_old,2] # arc length along path
        l_new = path_data[i_new,2]

        if d_old < wake_width: # close enough to wake, should move closer to the path
            # When moved toward path
            # the further it was, the more reward
            # the closer it moved, the more reward

            # When moved away from path
            # the closer it was, the more penalty
            # the faraway it moved, the more penalty
            ka = 2.0
            kc = 0.5
            reward = ka*(d_old - d_new)/(d_old+kc*wake_width)*wake_width

            # progress reward (agent should move forward)
            # gauge progress against L0
            kl = 400.0 # todo: more vigorous way to calculate this
            reward += (l_new - l_old)/L0*kl

        else: # no sufficient info to guide action, neutral reward
            reward = 0
    
        print(f'Step reward: d_old-d_new {d_old-d_new:.1f}, l_old-l_new {l_old-l_new:.1f}, reward {reward:.1f}')
        rewards.append(reward)
        vertical_distances.append(d_old-d_new)
        progres_distances.append(l_old-l_new)
    return rewards, vertical_distances, progres_distances


def test_env():
    path_xy = read_object_path_xy()
    path_data = calc_path_data(path_xy)

    print(path_data)

def visualization():
    path_xy = read_object_path_xy()
    path_data = calc_path_data(path_xy)
    RL_max_step = 100
    offset = 10
    random_offset = [random.randint(-offset, offset) for _ in range(RL_max_step)]
    variance = np.random.randint(-50, 50, size=(RL_max_step, 2))
    step_length = int(len(path_xy) / RL_max_step * 0.75)
    # print(step_length)
    idx_list = [a+b for a, b in zip(list(range(0,int(len(path_xy) * 0.75),step_length)[:RL_max_step]), random_offset)]
    data = path_xy[idx_list] + variance
    data = data[data[:,0]<1700]
    rewards, ver_diss, prgresses = calc_step_reward(loc_steps=data, path_data=path_data)
    cum_rewards = np.cumsum(rewards)

    # Create x-axis values (indices)    
    x_indices = np.arange(len(rewards))

    # Create the plot
    plt.figure(figsize=(8, 4))  # Optional: Adjust figure size

    # Plot the data as a line with dot markers
    # 'o-' specifies circular markers and a solid line connecting them
    plt.plot(x_indices, rewards, 'o-', color='red', label='rewards')
    # Add labels and title for clarity
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Step-wise reward")
    plt.grid(True)  # Optional: Add a grid for better readability

    # Display the plot
    plt.savefig('step_reward.png')   
    plt.close()

    plt.figure(figsize=(8, 4))  # Optional: Adjust figure size
    # Plot the data as a line with dot markers
    # 'o-' specifies circular markers and a solid line connecting them
    plt.plot(x_indices, cum_rewards, 'o-', color='red', label='cumulartive_reward')
    # Add labels and title for clarity
    plt.xlabel("Step")
    plt.ylabel("Sum_reward")
    plt.title("Cumulative reward")
    plt.grid(True)  # Optional: Add a grid for better readability

    # Display the plot
    plt.savefig('cum_reward.png')   
    plt.close()

    plt.figure(figsize=(8, 4))  # Optional: Adjust figure size
    plt.plot(x_indices, ver_diss, 'o-', color='blue', label='vertical_distances')
    plt.plot(x_indices, prgresses, 'o-', color='yellow', label='progress_distances')

    # Add labels and title for clarity
    plt.xlabel("Step")
    plt.ylabel("Distance values (mm)")
    plt.title("Step-wise evaluation")
    plt.grid(True)  # Optional: Add a grid for better readability

    # Display the plot
    plt.savefig('step_distances.png')

    plt.close()

    # Extract x and y coordinates from the array
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    # Plot the data points as dots
    plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Data Points')
    plt.scatter(path_xy[:,0], path_xy[:,1], color='yellow', marker='^', label='Data Points')
    # Plot the data as a line connecting the points
    plt.plot(x_coords, y_coords, color='red', linestyle='-', label='Connecting Line')

    plt.plot(path_xy[:,0], path_xy[:,1], color='green', linestyle='-', label='Object Line')

    # Add labels and title for clarity
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Plot of Dynamic Tracking")
    plt.legend()  # Display the legend

    # Show the plot
    plt.grid(True)  # Add a grid for better visualization
    plt.savefig('visual.png')

if __name__ == "__main__":
    # test1()
    visualization()