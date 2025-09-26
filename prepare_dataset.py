import pandas as pd

# Load both datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels: 1 = Fake, 0 = Real
fake['label'] = 1
true['label'] = 0

# Combine
df = pd.concat([fake, true], axis=0)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Save as dataset.csv for training
df.to_csv("dataset.csv", index=False)

print("✅ dataset.csv created with shape:", df.shape)
print(df.head())
