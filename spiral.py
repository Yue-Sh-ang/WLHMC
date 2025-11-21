
import numpy as np
import WLHMC_utils
import torch 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#data set
class SpiralDataset:
    def __init__(self, num_classes, num_data_per_class, rmin, rmax, dtheta_dr, noise, seed, device='cpu'):
        self.num_classes = num_classes
        self.num_data_per_class = num_data_per_class
        self.num_data = num_classes * num_data_per_class

        rng = np.random.default_rng(seed)

        # Allocate arrays
        self.x = torch.zeros((2, self.num_data), device=device)  # shape (2, N)
        self.y = torch.zeros((num_classes, self.num_data), device=device)  # one-hot encoding

        # Generate data
        for i in range(num_classes):
            for j in range(num_data_per_class):
                r = rng.uniform(rmin, rmax)
                theta = dtheta_dr * r + (2 * np.pi * i) / num_classes
                index = i * num_data_per_class + j

                self.x[0, index] = r * np.cos(theta) + rng.normal(0, noise)
                self.x[1, index] = r * np.sin(theta) + rng.normal(0, noise)
                self.y[i, index] = 1.0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
datax=SpiralDataset(num_classes=2, num_data_per_class=30, rmin=1, rmax=5, dtheta_dr=2, noise=0.1, seed=0,device=device)
datay=SpiralDataset(num_classes=2, num_data_per_class=30, rmin=1, rmax=5, dtheta_dr=2, noise=0.1, seed=1,device=device)


#network and loss function
class FCNetwork(nn.Module):
    def __init__(self, input_channel, channels):
        super().__init__()
        self.input_channel = input_channel
        self.channels = channels

        layers = []
        prev = input_channel
        for i, c in enumerate(channels):
            layers.append(nn.Linear(prev, c))
            if i != len(channels) - 1:
                layers.append(nn.ReLU())
            prev = c
        self.net = nn.Sequential(*layers)

        # Custom initialization: U[-3/sqrt(W), 3/sqrt(W)]
        for m in self.net:
            if isinstance(m, nn.Linear):
                W = m.in_features  # number of inputs (fan-in)
                bound = 2.0 / math.sqrt(W)
                nn.init.uniform_(m.weight, -bound, bound)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

net=FCNetwork(input_channel=2, channels=[8,8,8,2]).to(device) # hidden layer with 100 neurons?

def cal_loss(net, datax):
    y_hat=net(datax.x.T.float())
    y=torch.tensor(datax.y.T, dtype=torch.float32)
    
    eps = 1e-12
    return -(y * torch.log(y_hat + eps)).sum()

def ln_loss(net, datax):
    loss=cal_loss(net, datax)
    return torch.log(loss)

def accuracy(net, datax):
    y_hat = net(datax.x.T.float())
    preds = y_hat.argmax(dim=1)
    true_classes = datax.y.argmax(dim=0)
    return (preds == true_classes).float().mean().item()

# Boundary
boundary = []
for name, param in net.named_parameters():
    num_neuron = param.size()[0]
    para_bound = 3 / np.sqrt(num_neuron) 
    boundary.append(para_bound * torch.ones_like(param.data))

boundary = torch.cat([bound.view(-1) for bound in boundary]).to(device)


# entropy map,no bias
bin_begin=-15
bin_end=15
bin_width=0.05
margin=0
bin_num = int((bin_end-bin_begin)/bin_width+2*margin+2)
entropy = torch.zeros((bin_num, bin_num),device=device)

entropymap=WLHMC_utils.EntropyMap(entropy,bin_begin,bin_end,bin_begin,bin_end)

sampler=WLHMC_utils.Sampler_WLHMC(net, entropymap, ln_loss,ln_loss,datax,datay, device,boundary)


def choose_Scalefactor(epoch):
    global scale_factor
    maxWLFactor=1
    warmupStep=1000
    global ExploreStep
    ExploreStep=50000
    
    if epoch < warmupStep:
            scale_factor = maxWLFactor / warmupStep * epoch
    elif epoch <ExploreStep:
            scale_factor=maxWLFactor
    else:                       # factor should scale as 1/t
            scale_factor = maxWLFactor * ExploreStep*2 / (epoch+ExploreStep)
    
    return scale_factor


Tprint=1000
Tsave=50000
acc=0
logfile = "sample_log.txt"
with open(logfile, "a") as f:
     for i in range(1_000_001):  # run 1 million steps
        choose_Scalefactor(i)
        accept = sampler.sample(scale_factor, lr0=0.0005, L=20, digit=4)
        if accept:
            acc += 1
        
        if i % Tprint == 0:
            acc_rate = acc / Tprint
            acc = 0  # reset counter

            train_loss = sampler.y.cpu().item()
            val_loss = sampler.x.cpu().item()

            msg = (
                f"step: {i}, "
                f"train loss: {train_loss:.6f}, "
                f"val loss: {val_loss:.6f}, "
                f"acc: {acc_rate:.4f}\n"
            )
            f.write(msg)
            f.flush()  # important if job crashes
            print(msg, end="")
        if i % Tsave == 0:
            torch.save(net.state_dict(), f"net_{i}.pth")
            sampler.entropymap.save(f"entropy_{i}.npz")

