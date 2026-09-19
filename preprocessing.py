import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
hdf5_filename = os.path.join(base_dir, "elec292_project.h5")

members = ["Yaz", "Marcus", "Dave"]

columns_to_smooth = [
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)",
    "Absolute acceleration (m/s^2)"
]

for member in members:
    df = pd.read_hdf(hdf5_filename, key=f"Raw_data/{member}")

    original_signal = df["Linear Acceleration x (m/s^2)"].copy()

    # Fill missing values
    df = df.ffill().bfill()

    # Smooth the acceleration signals
    for column in columns_to_smooth:
        df[column] = df[column].rolling(window=5, center=True).mean()

    # Fill NaNs created by rolling
    df = df.ffill().bfill()

    # Plot original vs smoothed signal
    plt.figure()
    plt.plot(original_signal[:500], label="Original")
    plt.plot(df["Linear Acceleration x (m/s^2)"][:500], label="Smoothed")
    plt.title(f"{member} Acceleration (Noise Removal)")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()

    df.to_hdf(hdf5_filename, key=f"Pre-processed_data/{member}", mode="a")

print("Pre-processing complete.")