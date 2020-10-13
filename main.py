import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_train = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x, y):
        x = torch.cat((x,y), dim=1)
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x, y):
        x = torch.cat((x,y), dim=1)
        x = self.model(x)
        return x

def make_one_hot(labels, C=10):
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(device)
    target = one_hot.scatter_(1, labels.unsqueeze(1), 1)
    target = Variable(target)
    return target

batch_size=100
data_train = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator().to(device)
D = Discriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(D.parameters(), lr=0.0001)
criterion = nn.BCELoss()

start = time.time()
print("[+] Train Start")
total_epochs = 200
total_batch = len(data_train)
for epoch in range(total_epochs):
    avg_cost = [0, 0]
    for x, y in data_train:
        x = x.view(x.size(0), -1).to(device)
        x_oh = make_one_hot(y.to(device), 10)

        z = torch.randn(batch_size, 100, device=device)
        z_label = torch.randint(10, (batch_size,), device=device)
        z_oh = make_one_hot(label, 10)

        z_img = G(z, z_oh)

        real = (torch.FloatTensor(x.size(0), 1).fill_(1.0)).to(device)
        fake = (torch.FloatTensor(x.size(0), 1).fill_(0.0)).to(device)
        
        # Train Generator
        optim_G.zero_grad()
        g_cost = criterion(D(z_img, z_oh), real)
        g_cost.backward()
        optim_G.step()

        z_img = z_img.detach().to(device)
        # Train Discriminator
        optim_D.zero_grad()
        d_cost = criterion(D(torch.cat((x, z_img)), torch.cat((x_oh, z_oh))), torch.cat((real, fake)))
        d_cost.backward()
        optim_D.step()

        avg_cost[0] += g_cost
        avg_cost[1] += d_cost
    
    avg_cost[0] /= total_batch
    avg_cost[1] /= total_batch

    if (epoch+1) % 10 == 0 or epoch < 10:
      print("Epoch: %d, Generator: %f, Discriminator: %f"%(epoch+1, avg_cost[0], avg_cost[1]))
      z = torch.randn(batch_size, 100, device=device)
      label = torch.Tensor(100).fill_(0).long().to(device)
      for i in range(10):
        for j in range(10):
          label[10*i+j] = j
      z_oh = make_one_hot(label, 10)
      z_img = G(z, z_oh)
      z_img = z_img.reshape([batch_size, 1, 28, 28])
      img_grid = make_grid(z_img, nrow=10, normalize=True)
      save_image(img_grid, "result/%d.png"%(epoch+1))

end = time.time()
total = int(end-start)
print("[+] Train Time : %dm %ds"%(total//60, total%60))