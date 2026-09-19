import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(os.path.abspath(__file__))
hdf5_filename = os.path.join(base_dir, "elec292_project.h5")

# Load segmented data
train_df = pd.read_hdf(hdf5_filename, key="Segmented_data/Train")
test_df = pd.read_hdf(hdf5_filename, key="Segmented_data/Test")

# Separate features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Convert labels to numeric
y_train = y_train.map({"walking": 0, "jumping": 1})
y_test = y_test.map({"walking": 0, "jumping": 1})

# Logistic regression using SGD
model = SGDClassifier(loss="log_loss")

epochs = 20
training_accuracy = []

for i in range(epochs):
    model.partial_fit(X_train, y_train, classes=[0, 1])

    predictions = model.predict(X_train)
    acc = accuracy_score(y_train, predictions)

    training_accuracy.append(acc)

# Plot training curve
plt.plot(training_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Curve (Logistic Regression)")
plt.show()

# Test set evaluation
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Test Accuracy:", test_accuracy)

joblib.dump(model, os.path.join(base_dir, "activity_model.pkl"))
print("Saved model to activity_model.pkl")
