import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier

# 📥 Load data
df = pd.read_csv(r'C:\laragon\www\skripsi\EEG FEATURES\new3\features_all_commands.csv')

# 🎯 Extract features and labels
X = df.select_dtypes(include=[np.number])
y_raw = df['label'] if 'label' in df.columns else df['command']

# 🔢 Encode string labels into integers for XGBoost compatibility
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)  # encoded labels
class_names = label_encoder.classes_     # keep original class names

# ⚖️ Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🧪 Cross-validation setup
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=50)

def evaluate_model(model, model_name="Model"):
    print(f"\n🔍 Evaluating: {model_name}")
    accuracies = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), start=1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        print(f"  Fold {fold} Accuracy: {acc:.2f}")

    print(f"\n✅ Cross-validation accuracy (mean ± std): {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
    print("\n📄 Classification Report:\n", classification_report(
        label_encoder.inverse_transform(y_true_all),
        label_encoder.inverse_transform(y_pred_all),
        target_names=class_names
    ))

    return model, y_true_all, y_pred_all

# 🌲 Evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model, rf_true, rf_pred = evaluate_model(rf_model, "Random Forest")

# 🌿 Evaluate XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model, xgb_true, xgb_pred = evaluate_model(xgb_model, "XGBoost")

# 📊 Confusion Matrix - XGBoost
conf_mat = confusion_matrix(xgb_true, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.title('📉 Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# 📈 Feature Importance - XGBoost
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=np.array(X.columns)[indices])
plt.title("📌 Top Feature Importances - XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
