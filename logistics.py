import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
df = pd.read_excel(r"F:\research paper\Retail-Supply-Chain-Sales-Dataset.xlsx")

# Convert date column to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
df.set_index("Order Date", inplace=True)  # Set it as index if needed

# Display sample data
print("Sample Data:\n", df.head())

# Step 2: Preprocessing 
# Selecting features
features = ["Quantity"]  # Only using sales quantity for now
target = "Quantity"

# Normalize data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Create sequences for Transformer model
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i : i + seq_length][features].values
        label = data.iloc[i + seq_length][target]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 10  # Looking back 10 days
X, y = create_sequences(df, seq_length)

# Split data into train and test sets (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Step 3: Transformer-Based Model 
class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_size, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take last time step's output
        return self.fc(x).squeeze()

# Model parameters
feature_size = len(features)
num_heads = 1
num_layers = 3
dropout = 0.1

model = TransformerTimeSeries(feature_size, num_heads, num_layers, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training the Model 
epochs = 50
batch_size = 32
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Step 5: Model Evaluation 
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# Convert back to original scale
y_test = y_test.numpy().reshape(-1, 1)
predictions = predictions.reshape(-1, 1)

# Step 6: Visualization 
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Demand", color="blue")
plt.plot(predictions, label="Predicted Demand", linestyle="dashed", color="red")
plt.legend()
plt.title("Transformer-Based Demand Forecasting on Real Data")
plt.xlabel("Time")
plt.ylabel("Quantity Sold")
plt.show()

