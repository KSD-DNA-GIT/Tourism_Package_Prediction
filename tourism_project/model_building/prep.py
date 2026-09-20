import pandas as pd
from sklearn.model_selection import train_test_split

# Load validated data
df = pd.read_csv('tourism_project/data/tourism_validated.csv')

# Remove non-predictive identifier
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# Separate features and target
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified Train-Test Split (80/20) keeps the purchase imbalance consistent
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits for the training phase
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Data Preparation complete. Artifacts saved.")
