import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv(r'C:\laragon\www\skripsi\EEG PROCESSED\commands\cca_bandpower_features.csv')
X = df[['corr_landing', 'corr_terbang', 'corr_kanan', 'corr_kiri', 'corr_depan', 'corr_belakang']]
y = df['command']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    "SVM (RBF)": SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n🔍 Evaluating: {name}")
    accuracies = []
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Metrics
    print(f"Cross-validation accuracy (mean ± std): {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
    print("Classification Report:\n", classification_report(y_true_all, y_pred_all))

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
