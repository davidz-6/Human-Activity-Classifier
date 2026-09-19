import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
hdf5_filename = os.path.join(base_dir, "elec292_project.h5")

members = ["Yaz", "Marcus", "Dave"]

# Use a different colormap
cmap = plt.cm.viridis
colors = cmap(np.linspace(0,1,3))

for member in members:

    df = pd.read_hdf(hdf5_filename, key=f"Raw_data/{member}")

    walk_df = df[df["label"] == "walking"].iloc[:500].reset_index(drop=True)
    jump_df = df[df["label"] == "jumping"].iloc[:500].reset_index(drop=True)

    axes_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]

    fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)

    for i, axis in enumerate(axes_cols):

        axes[i].plot(walk_df[axis], color=colors[0], label="Walking")
        axes[i].plot(jump_df[axis], color=colors[2], label="Jumping")

        axes[i].set_title(f"{member} {axis}")
        axes[i].set_ylabel("Acceleration (m/s²)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("Sample")

    plt.suptitle(f"{member} Accelerometer Signals")
    plt.tight_layout()
    plt.show()



    walk_values = walk_df[axes_cols].values.flatten()
    jump_values = jump_df[axes_cols].values.flatten()

    plt.figure(figsize=(7,5))

    plt.boxplot(
        [walk_values, jump_values],
        patch_artist=True,
        boxprops=dict(facecolor=colors[1]),
        medianprops=dict(color="black")
    )

    plt.xticks([1,2], ["Walking", "Jumping"])
    plt.ylabel("Acceleration (m/s²)")
    plt.title(f"{member} Acceleration Distribution")
    plt.grid(True, axis="y", alpha=0.3)

    plt.show()
