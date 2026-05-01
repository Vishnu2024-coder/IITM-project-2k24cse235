import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load dataset
# -------------------------------
train = pd.read_csv("data/KDDTrain+.txt", header=None)
test = pd.read_csv("data/KDDTest+.txt", header=None)

# -------------------------------
# 2. Column names
# -------------------------------
columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins",
"logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label","difficulty"
]

train.columns = columns
test.columns = columns

train = train.drop("difficulty", axis=1)
test = test.drop("difficulty", axis=1)

# -------------------------------
# 3. Encode categorical
# -------------------------------
cat_cols = ["protocol_type", "service", "flag"]

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]])
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# -------------------------------
# 4. Labels
# -------------------------------
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# -------------------------------
# 🔥 5. ADD ADVERSARIAL FEATURES
# -------------------------------

# Feature 1: perturbation score (input inconsistency)
train['perturbation_score'] = np.abs(train['src_bytes'] - train['dst_bytes']) / (train['src_bytes'] + 1)
test['perturbation_score'] = np.abs(test['src_bytes'] - test['dst_bytes']) / (test['src_bytes'] + 1)

# Feature 2: traffic anomaly ratio (helps adversarial evasion detection)
train['traffic_ratio'] = train['count'] / (train['srv_count'] + 1)
test['traffic_ratio'] = test['count'] / (test['srv_count'] + 1)

# -------------------------------
# 6. Split
# -------------------------------
X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']

# -------------------------------
# 7. Normalize
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 8. Handle imbalance
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------------
# 9. Train model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# -------------------------------
# 🔥 10. Adversarial Feature: Prediction Uncertainty
# -------------------------------
probs = model.predict_proba(X_test)
uncertainty = 1 - np.max(probs, axis=1)

# Add uncertainty as a feature for analysis (not retraining here)
X_test_with_uncertainty = np.hstack((X_test, uncertainty.reshape(-1,1)))

# -------------------------------
# 11. Test
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -------------------------------
# 12. Show adversarial detection insight
# -------------------------------
print("\nSample Uncertainty Scores (Adversarial Indicator):")
print(uncertainty[:10])