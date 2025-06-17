import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Step 1: Load and Preprocess Data
df = pd.read_csv("dataset/KFD_399.csv")

# Convert Y/N to 1/0
yn_columns = df.columns[3:-1]
for col in yn_columns:
    df[col] = df[col].map({'Y': 1, 'N': 0})

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Season', 'GPS_loc', 'Occupation'], drop_first=True)

# Convert multi-class labels
df['KFD'] = df['KFD'].map({'C': 0, 'S': 1, 'N': 2, 'PR': 3})
df.dropna(subset=['KFD'], inplace=True)

# Step 2: Feature Selection
X = df.drop('KFD', axis=1)
y = df['KFD']

def feature_selection(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features], selected_features

X_selected, selected_features = feature_selection(X, y, k=10)

# Step 3: Scaling and Splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Step 5: Evaluate (Optional)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_svm = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# Step 6: Save the Model, Scaler, and Selected Features
joblib.dump(svm_model, 'Models/best_kfd_model.pkl')
joblib.dump(scaler, 'Models/scaler.pkl')
joblib.dump(selected_features, 'Models/selected_features.pkl')  # Save selected features

print("SVM model, scaler, and selected features saved successfully!")