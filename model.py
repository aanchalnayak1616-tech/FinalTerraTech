import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Define file paths
original_csv = 'Crop_recommendation.csv'

# Check if CSV exists
if not os.path.exists(original_csv):
    raise FileNotFoundError(f"CSV file '{original_csv}' not found in directory: {os.getcwd()}")

# Load the original CSV for training
try:
    df_original = pd.read_csv(original_csv)
except Exception as e:
    raise Exception(f"Failed to load '{original_csv}': {str(e)}")

# Define required columns
required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Check if original CSV has the required columns and label
if not all(col in df_original.columns for col in required_columns):
    raise ValueError(f"CSV file must contain the required columns: {required_columns}")
if 'label' not in df_original.columns:
    raise ValueError("CSV file must contain a 'label' column for training.")

# Prepare data for training
X = df_original[required_columns]
y = df_original['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.4f}")

# Save the model and scaler
try:
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(sc, open("scaler.pkl", "wb"))
    print(f"Model and scaler saved as model.pkl and scaler.pkl in {os.getcwd()}")
except Exception as e:
    raise Exception(f"Failed to save model.pkl or scaler.pkl: {str(e)}")

# Example usage: Predict crop for a single set of features
example_features = [[90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362]]  # Example from CSV
example_features_scaled = sc.transform(example_features)
predicted_crop = model.predict(example_features_scaled)
print(f"Predicted crop for example features: {predicted_crop[0]}")