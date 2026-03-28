import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)

x = 2 * np.random.rand(100,1)
y = 1 + 2 * x + np.random.rand(100,1)

# Convert numpy to Tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearRegression()
cost_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

# Training Loop
epochs = 1000

for epoch in range(epochs):
    outputs = model(x_tensor)
    loss = cost_function(outputs, y_tensor)

    # Backprop pass e optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: [{epoch + 1} / {epochs}], Loss: {loss.item():.4f}")


# Save weigths
torch.jit.save(torch.jit.script(model),'model.pt')
