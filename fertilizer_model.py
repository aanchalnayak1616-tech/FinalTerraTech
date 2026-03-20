import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("india_soil_fertilizer.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Normalize text data
df['soil_type'] = df['soil_type'].str.lower().str.strip()
df['crop_type'] = df['crop_type'].str.lower().str.strip()
df['fertilizer'] = df['fertilizer'].str.lower().str.strip()

# -------------------------------
# Encode categorical data
# -------------------------------
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

df['soil_type'] = soil_encoder.fit_transform(df['soil_type'])
df['crop_type'] = crop_encoder.fit_transform(df['crop_type'])
df['fertilizer'] = fert_encoder.fit_transform(df['fertilizer'])

# -------------------------------
# Features & Target
# -------------------------------
features = ['soil_type','crop_type','soil_pH','nitrogen','phosphorus','potassium']

X = df[features]
y = df['fertilizer']

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X, y)

print("✅ Fertilizer model trained")

# -------------------------------
# Save model & encoders
# -------------------------------
pickle.dump(model, open("fertilizer_model.pkl", "wb"))
pickle.dump(soil_encoder, open("soil_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))
pickle.dump(fert_encoder, open("fert_encoder.pkl", "wb"))

print("✅ All files saved successfully")