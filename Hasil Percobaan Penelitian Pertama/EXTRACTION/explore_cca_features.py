import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
file_path = r'C:\laragon\www\skripsi\EEG PROCESSED\commands\cca_features_multiclass.csv'
df = pd.read_csv(file_path)

# Use 'command' as label
label_col = 'command'

# 1. Show shape and preview
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# 2. Data types and basic info
print("\nDataset Info:")
print(df.info())

# 3. Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# 4. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Check class distribution
if label_col in df.columns:
    print("\nClass Distribution:")
    print(df[label_col].value_counts())

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=label_col, data=df, palette='Set2')
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"\n⚠️ '{label_col}' column not found. Please check column name.")

# 6. Correlation Heatmap (only for numeric features)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 7. PCA for Visualization (2D)
features = numeric_df.drop(columns=['loop'], errors='ignore')  # drop 'loop' if not needed
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca[label_col] = df[label_col]

# Plot PCA result
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue=label_col, palette='Set2', s=100, edgecolor='k')
plt.title("PCA - 2D Visualization of Commands")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 8. Histograms of features
features.hist(figsize=(16, 12), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()
