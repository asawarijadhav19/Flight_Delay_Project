import joblib

features = joblib.load("model_features.pkl")

print("MODEL FEATURES:")
for feature in features:
    print(feature)