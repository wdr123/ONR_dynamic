from whisker_driver import WhiskerArraySimEnv
from whisker_driver import read_object_path_xy
from whisker_driver import calc_path_data
import random
import numpy as np
import matplotlib.pyplot as plt


def test_env():
    path_xy = read_object_path_xy()
    path_data = calc_path_data(path_xy)

    print(path_data)

def visualization():
    path_xy = read_object_path_xy()
    path_data = calc_path_data(path_xy)
    RL_max_step = 100
    offset = 0
    random_offset = [random.randint(-offset, offset) for _ in range(RL_max_step)]
    step_length = int(len(path_xy) / RL_max_step * 0.75)
    # print(step_length)
    idx_list = [a+b for a, b in zip(list(range(0,int(len(path_xy) * 0.75),step_length)[:RL_max_step]), random_offset)]
    # print(path_xy[idx_list])


    # Extract x and y coordinates from the array
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    # Plot the data points as dots
    plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Data Points')

    # Plot the data as a line connecting the points
    plt.plot(x_coords, y_coords, color='red', linestyle='-', label='Connecting Line')

    # Add labels and title for clarity
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Plot of NumPy Array with Line and Dots")
    plt.legend()  # Display the legend

    # Show the plot
    plt.grid(True)  # Add a grid for better visualization
    plt.show()


if __name__ == "__main__":
    # test1()
    visualization()