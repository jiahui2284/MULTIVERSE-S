import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


# ===== Get the USD/EUR dataset=====
df = yf.download("USDEUR=X", start="2025-01-01", interval="1d")
df = df.reset_index()[["Date", "Close"]].rename(columns={"Close": "Price"}).sort_values("Date")

print(df.tail())  # view recent data

# ===== Normlization Data=====
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

# ===== Make the sample ,use past 7 days  =====
X, y = [], []
for i in range(7, len(scaled_prices)):
    X.append(scaled_prices[i-7:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, 60, 1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (samples, 1)

# ===== LSTM model =====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # The last time step

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== Training Model  =====
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20, Loss: {loss.item():.6f}")

# ===== Preict next days  =====
model.eval()
last_7 = scaled_prices[-7:]                               # (7, 1)
last_7 = torch.tensor(last_7, dtype=torch.float32).unsqueeze(0)  # (1, 7, 1)

with torch.no_grad():
    pred_scaled = model(last_7).numpy()

pred_price = scaler.inverse_transform(pred_scaled)
print("\nPredicted next-day EUR/USD exchange rate:", float(pred_price[0][0]))
