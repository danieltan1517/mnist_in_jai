import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy

batch_size = 225
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.layer1 = nn.Linear(784, 512) #dtype=int8
    self.layer2 = nn.Linear(512, 10)

  def forward(self, x):
    x = relu(self.layer1(x))
    x = self.layer2(x)
    return x

def accuracy(loader, model):
  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.reshape(x.shape[0], -1)

      scores = model(x)
      _, predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")


device = None
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

input_size = 784
num_classes = 10
learning_rate = 0.001
num_epochs = 20

neural_network = NN()
model = neural_network.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  print(f"Epoch: {epoch} out of {num_epochs}")
  for i, (data, targets) in enumerate(train_loader):
    data = data.to(device=device)
    targets = targets.to(device=device)

    # flatten all outer dimensions into one.
    # 28x28 => 784
    data = data.reshape(data.shape[0], -1)

    # forward propagation.
    scores = model(data)
    loss   = criterion(scores, targets)

    # zero previous gradients
    optimizer.zero_grad()

    # backpropagation.
    loss.backward()

    # gradient descent or adam step.
    optimizer.step()

accuracy(train_loader, model)
accuracy(test_loader, model)

with open('mnist.nn', 'wb') as nn_file:
  nn_file.write(b'MNIST_MODEL')
  for params in model.parameters():
    tensor = params.data.cpu()
    requires_grad = params.requires_grad
    print(tensor.shape)
    flattened = torch.flatten(tensor)
    numpy_array = flattened.numpy()
    data_bytes = numpy_array.tobytes()
    nn_file.write(data_bytes)



