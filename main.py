import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.join(base_dir, "292 3")
hdf5_filename = os.path.join(base_dir, "elec292_project.h5")

member_data = {
    "Yaz": [],
    "Marcus": [],
    "Dave": []
}

for file_name in os.listdir(root_directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(root_directory, file_name)
        df = pd.read_csv(file_path)

        parts = os.path.splitext(file_name)[0].split()
        action_code = parts[0]
        member_code = parts[1]

        if member_code == "Y":
            member = "Yaz"
        elif member_code == "M":
            member = "Marcus"
        else:
            member = "Dave"

        if action_code.startswith("W"):
            df["label"] = "walking"
        else:
            df["label"] = "jumping"

        member_data[member].append(df)

for member in member_data:
    combined_df = pd.concat(member_data[member], ignore_index=True)
    combined_df.to_hdf(hdf5_filename, key=f"Raw_data/{member}", mode="a")

