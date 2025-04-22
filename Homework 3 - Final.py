import requests
import gzip
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnhelper import eval_model

class CustomMNIST(Dataset):
    def __init__(self, image_url, label_url):
        self.images = self.read_images(image_url)
        self.labels = self.read_labels(label_url)
                
    def download_file(self, url):
        print(url)
        response = requests.get(url)
        print(response)
        compressed_file = io.BytesIO(response.content)
        return compressed_file
    
    def read_labels(self,label_url):
        compressed_file = self.download_file(label_url)
        #LLM assisted with logic below that reads the labels file
        with gzip.open(compressed_file, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')  #skipping 4-byte integer
            _ = int.from_bytes(f.read(4), 'big')  #skipping 4-byte integer
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def read_images(self,image_url):
        compressed_file = self.download_file(image_url)
        #LLM assisted with logic below that reads the image file
        with gzip.open(compressed_file, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 1, rows, cols)
        return images

    # return the length of the complete data set
    def __len__(self):
        return self.images.shape[0]
    
    # retrieve a single record based on index position `idx`
    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32) #convert to tensor object
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long) #convert to tensor object
        return image, label

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.LazyConv2d(6, 5, padding=2)
        self.conv2 = nn.LazyConv2d(16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.LazyLinear(120)  
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    # Loop over batches via the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and looking for improved gradients
        loss.backward()
        optimizer.step()
        # Zeroing out the gradient (otherwise they are summed)
        #   in preparation for next round
        optimizer.zero_grad()

        # Print progress update every few loops
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_net(model, train_dataloader, test_dataloader, epochs=5, learning_rate=1e-3):
    lr = learning_rate
    ep = epochs
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    
    for t in range(ep):
        try:
            print(f"Epoch {model.EPOCH+t+1}\n-------------------------------")
        except:
            print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    try:
        model.EPOCH += ep
    except:
        model.EPOCH = ep
    return model

def save_model(epochs, model,PATH,optimizer):
    # Save model as .pt file
    EPOCH = epochs
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)

def main():  
    train_dataset = CustomMNIST('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz','https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz')
    test_dataset = CustomMNIST('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz','https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz')
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    PATH = "nn-model-04212025.pt"
    learning_rate = 1e-3
    epochs = 5 
    
    #I just used VS Code on my very old ASUS laptop, so I did not use GPUs
    model = LeNet()#.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    model = train_net(model, train_dataloader,test_dataloader,epochs,learning_rate)

    save_model(epochs, model,PATH,optimizer)
    eval_model(PATH, LeNet(), test_dataloader)

main()