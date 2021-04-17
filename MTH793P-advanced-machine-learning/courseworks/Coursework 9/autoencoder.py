import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fct
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode_1 = nn.Linear(784, 100)
        self.encode_2 = nn.Linear(100, 32)
        self.decode_1 = nn.Linear(32, 100)
        self.decode_2 = nn.Linear(100, 784)
        self.relu = nn.ReLU()
    
    def encoder(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        h1 = self.relu(self.encode_1(inputs))
        h2 = self.encode_2(h1)
        return h2
    
    def decoder(self, codes):
        # YOUR CODE HERE
        h3 = self.relu(self.decode_1(codes))
        h4 = self.decode_2(h3)
        output = h4.reshape(-1, 1, 28, 28)
        return output
    
    def forward(self, inputs):        
        return self.decoder(self.encoder(inputs))


def train_loop(loader, model, loss_fn, optimizer):
    size = len(loader)
    for batch, (X, y) in enumerate(loader):
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")


def train_model(loader, model, loss_fn, optimizer, epochs=20):
    for e in range(epochs):
        print(f"Epoch {e+1}\n{'*' * 30}")
        train_loop(loader, model, loss_fn, optimizer)
    model.eval()


def main():
    torch.set_default_tensor_type(torch.FloatTensor)
    training_batch_size = 150
    test_batch_size = 150

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [1])
                ])), 
            batch_size=training_batch_size, shuffle=True, num_workers=4)

    autoencoder = Autoencoder()
    loss = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    train_model(train_loader, autoencoder, loss, optimizer)
    torch.save(autoencoder.state_dict(), "autoencoder")



if __name__ == "__main__":
    main()