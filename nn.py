from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x
    
model = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X = np.load('X.npy')
y = np.load('y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

epochs = 300

losses = []

for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    y_pred_classes = y_pred.argmax(dim=1)
    correct = (y_pred_classes == y_train).sum()

    accuracy = correct / y_train.shape[0]

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_eval = model(X_test)

    y_eval_classes = y_eval.argmax(dim=1)

    correct = (y_eval_classes == y_test).sum()
    accuracy = correct / y_test.shape[0]

    print(accuracy.item())

torch.save(model.state_dict(), "tictac_model.pth")
print("Model saved!")

