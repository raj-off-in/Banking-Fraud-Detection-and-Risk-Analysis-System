
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display the first five transactions
print("First five transactions:")
print(df.head())

# Check dataset size
print("\nDataset shape:")
print(df.shape)

# Check column names and data types
print("\nDataset information:")
df.info()

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Count legitimate and fraudulent transactions
print("\nTransaction class distribution:")
print(df["Class"].value_counts())

# Display the percentage of fraudulent transactions
fraud_percentage = (df["Class"].mean()) * 100
print(f"\nFraud percentage: {fraud_percentage:.3f}%")

# Visualize the transaction distribution
sns.countplot(data=df, x="Class")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()
