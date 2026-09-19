import pandas as pd

df = pd.read_hdf("elec292_project.h5", key="Pre-processed_data/Yaz")
print(df.columns.tolist())