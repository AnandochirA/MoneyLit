import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv('D:/ML/Assets/first/cleaned.csv')
X = df.drop('Value', axis=1).values
y = df['Value'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class StateCostMLP(nn.Module):
    def __init__(self):
        super(StateCostMLP, self).__init__()
        self.layer1 = nn.Linear(55, 32)  # Input: 55 features, Hidden: 32
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)  # Hidden: 32 -> 16
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)   # Output: 1 (Value)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

model = StateCostMLP()

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Average loss per epoch
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    if (epoch + 1) % 100 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 5. Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    print(f'Mean Absolute Error on Test Set: ${mae:.2f}')

# 6. Inflation Adjustment (2015 -> 2025, ~3% annual)
inflation_factor = 1.03 ** 10 
y_pred_2025 = y_pred * inflation_factor
print(f'Sample Prediction (2025): ${y_pred_2025[0].item():.2f}')

# 7. Plot Loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('StateCost MLP Training Loss')
plt.show()

# 8. Save Model
torch.save(model.state_dict(), 'statecost_mlp.pth')
print("Model saved as 'statecost_mlp.pth'")

custom_input = np.zeros(55)  # 55 features (0–54)
custom_input[0] = 2020   # Year
custom_input[48] = 1     # state_Virginia
custom_input[53] = 1     # type_Public Out-of-State
custom_input[54] = 1     # length_4-year

custom_input_scaled = scaler.transform([custom_input])
custom_tensor = torch.tensor(custom_input_scaled, dtype=torch.float32)
inflation_factor = 1.03 ** 5
with torch.no_grad():
    pred = model(custom_tensor)
    pred_2025 = pred.item() * inflation_factor
    print(f"Predicted 2025 Tuition (Virginia, Public Out-of-State, 4-year): ${pred_2025:.2f}")
    print(f"Total with $14K Room/Board: ${pred_2025 + 14000:.2f}")