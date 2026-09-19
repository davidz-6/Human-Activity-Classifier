import os
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

WINDOW_SIZE = 500  # 5-second windows

AXES = [
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)"
]

LABEL_MAP = {0: "walking", 1: "jumping"}


def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # fill missing values
    df = df.ffill().bfill()

    missing_cols = [col for col in AXES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    all_features = []
    window_ids = []

    num_windows = len(df) // WINDOW_SIZE

    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        group = df.iloc[start:end]

        features = {}

        for axis in AXES:
            features[f"{axis}_mean"] = group[axis].mean()
            features[f"{axis}_max"] = group[axis].max()
            features[f"{axis}_min"] = group[axis].min()
            features[f"{axis}_var"] = group[axis].var()
            features[f"{axis}_median"] = group[axis].median()

        all_features.append(features)
        window_ids.append(i + 1)

    if not all_features:
        raise ValueError("CSV is too short. Need at least one full window of data.")

    X = pd.DataFrame(all_features)
    return X, window_ids


class ActivityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Walking vs Jumping Classifier")
        self.root.geometry("560x320")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "activity_model.pkl")
        self.csv_path = None

        title = tk.Label(
            root,
            text="ELEC 292 Activity Classifier",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        self.status_label = tk.Label(
            root,
            text="No file selected.",
            wraplength=500,
            justify="center"
        )
        self.status_label.pack(pady=10)

        select_btn = tk.Button(
            root,
            text="Select Input CSV",
            command=self.select_csv,
            width=22
        )
        select_btn.pack(pady=5)

        run_btn = tk.Button(
            root,
            text="Classify Dataset",
            command=self.run_classification,
            width=22
        )
        run_btn.pack(pady=5)

        self.result_label = tk.Label(
            root,
            text="Overall result will appear here.",
            font=("Arial", 13, "bold"),
            fg="blue",
            wraplength=500,
            justify="center"
        )
        self.result_label.pack(pady=20)

        quit_btn = tk.Button(root, text="Exit", command=root.quit, width=22)
        quit_btn.pack(pady=5)

    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            self.status_label.config(text=f"Selected file:\n{path}")
            self.result_label.config(text="Overall result will appear here.", fg="blue")

    def run_classification(self):
        try:
            if not self.csv_path:
                messagebox.showwarning("No file", "Please select a CSV file first.")
                return

            if not os.path.exists(self.model_path):
                messagebox.showerror("Missing model", "activity_model.pkl not found.")
                return

            model = joblib.load(self.model_path)

            X, window_ids = extract_features_from_csv(self.csv_path)

            preds = model.predict(X)
            pred_labels = [LABEL_MAP[int(p)] for p in preds]

            # Save window-by-window predictions
            output_df = pd.DataFrame({
                "window": window_ids,
                "label": pred_labels
            })

            output_path = os.path.splitext(self.csv_path)[0] + "_predictions.csv"
            output_df.to_csv(output_path, index=False)

            # Majority vote for whole dataset
            walking_count = pred_labels.count("walking")
            jumping_count = pred_labels.count("jumping")

            if walking_count >= jumping_count:
                overall_label = "WALKING"
                result_color = "green"
            else:
                overall_label = "JUMPING"
                result_color = "red"

            self.status_label.config(
                text=f"Classification complete.\nSaved output to:\n{output_path}"
            )

            self.result_label.config(
                text=(
                    f"Overall result: {overall_label}\n"
                    f"Walking windows: {walking_count}\n"
                    f"Jumping windows: {jumping_count}"
                ),
                fg=result_color
            )

            messagebox.showinfo(
                "Success",
                f"Overall dataset classification: {overall_label}\n\n"
                f"Walking windows: {walking_count}\n"
                f"Jumping windows: {jumping_count}\n\n"
                f"Predictions saved to:\n{output_path}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityApp(root)
    root.mainloop()