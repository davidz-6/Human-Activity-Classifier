import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

hdf5_filename = "elec292_project.h5"
members = ["Yaz", "Marcus", "Dave"]

window_size = 500

all_features = []
all_labels = []

for member in members:
    df = pd.read_hdf(hdf5_filename, key=f'Pre-processed_data/{member}')

    df['segment_id'] = df.index // window_size
    grouped = df.groupby('segment_id')

    for seg_id, group in grouped:
        if len(group) < (window_size * 0.9):
            continue

        label = group['label'].iloc[0]

        features = {}
        axes = [
            'Linear Acceleration x (m/s^2)',
            'Linear Acceleration y (m/s^2)',
            'Linear Acceleration z (m/s^2)'
        ]

        for axis in axes:
            features[f'{axis}_mean'] = group[axis].mean()
            features[f'{axis}_max'] = group[axis].max()
            features[f'{axis}_min'] = group[axis].min()
            features[f'{axis}_var'] = group[axis].var()
            features[f'{axis}_median'] = group[axis].median()

        all_features.append(features)
        all_labels.append(label)

X = pd.DataFrame(all_features)
y = pd.Series(all_labels)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled['label'] = y.values

train_df, test_df = train_test_split(
    X_scaled, test_size=0.10, shuffle=True, random_state=42
)

train_df.to_hdf(hdf5_filename, key='Segmented_data/Train', mode='a')
test_df.to_hdf(hdf5_filename, key='Segmented_data/Test', mode='a')

print(f"Extracted {len(X.columns)} features from {len(X_scaled)} total 5-second windows.")
print(f"Training set size: {len(train_df)} windows")
print(f"Testing set size: {len(test_df)} windows")
print("Saved successfully to Segmented_data/Train and Segmented_data/Test!")

