import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import torch 
import sys
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import random
import pickle
import scipy.stats
from scipy.signal import convolve2d, fftconvolve
import math
import WHMC_utils

#data set
shuffled_df = pd.read_csv('../data/train_shuffled.csv')
X_train = shuffled_df.iloc[:, :-1]
y_train= shuffled_df.iloc[:, -1]
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
y_train = torch.tensor(y_train.values).float().to(device).unsqueeze(1) # Unsqueeze to match the shape of the output of our model
X_train = torch.tensor(X_train.values).float().to(device)

data_X=[X_train[0:len(X_train)//2],y_train[0:len(X_train)//2]]
data_Y=[X_train[len(y_train)//2:],y_train[len(y_train)//2:]]


# network and boundary
class net(nn.Module):
    def __init__(self, D_in=331, H=100, D_out=1, Hn=1):
        super().__init__()
        self.Hn = Hn # Number of hidden layer
        self.activation = nn.ReLU()
        #self.activation = nn.Softplus() # Activation function
        
        self.layers = nn.ModuleList([nn.Linear(D_in, H), self.activation]) # First hidden layer
        for i in range(self.Hn - 1):
            self.layers.extend([nn.Linear(H, H), self.activation]) # Add hidden layer
        self.layers.append(nn.Linear(H, D_out)) # Output layer
        self.init_weights()  # Initialize weights using the init_weights function
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, mean=0, std=0.05)
                init.constant_(layer.bias, 0)
                
net = net().to(device)
boundary=[]
for name, param in net.named_parameters():
        num_neuron = param.size()[0]
        para_bound = 3 / np.sqrt(num_neuron) 
        boundary.append(para_bound*torch.ones_like(param.data))

boundary = torch.cat([bound.view(-1) for bound in boundary])




#loss function
loss = nn.MSELoss(reduction='mean')
def cal_loss(net,data):
    X=data[0]
    Y=data[1]
    
    Yhat=net(X)

    return loss(Yhat, Y.reshape(Yhat.shape))
    
    
#entropy map initialization
bin_begin=-20
bin_end=20
bin_width=0.02
margin=0
bin_num = int((bin_end-bin_begin)/bin_width+2*margin+2)
entropy = np.zeros((bin_num, bin_num))
binEdges=(np.array(range(bin_num))-margin)*bin_width+bin_begin
xx, yy=np.meshgrid(binEdges, binEdges)
maxLogTrainLossToStudy = -1.0
maxLogTestLossToStudy = 3.0
forbiddenRegionEntropyIncrease = 1000

entropyBias = forbiddenRegionEntropyIncrease*(yy>maxLogTrainLossToStudy)*(yy-maxLogTrainLossToStudy)
entropyBias += forbiddenRegionEntropyIncrease*(xx>maxLogTestLossToStudy)*(xx-maxLogTestLossToStudy)
entropy=entropy+entropyBias

entropy = torch.as_tensor(entropy, dtype=torch.float32, device=device)
entropymap=WHMC_utils.EntropyMap(entropy,bin_begin,bin_end,bin_begin,bin_end)


#establish the sampler
Sampler=WHMC_utils.Sampler_WLHMC(net,entropymap,cal_loss,data_X,data_Y,device,boundary=boundary)
#need to take mre care of device

# scale factor
def choose_Scalefactor(epoch):
    global scale_factor
    maxWLFactor=10
    warmupStep=5000
    global ExploreStep
    ExploreStep=100000
    
    if epoch < warmupStep:
            scale_factor = maxWLFactor / warmupStep * epoch
    elif epoch <ExploreStep:
            scale_factor=maxWLFactor
    else:                       # factor should scale as 1/t
            scale_factor = maxWLFactor * ExploreStep*2 / (epoch+ExploreStep)
    
    return scale_factor

#main
for i in range(200001):
    dS=choose_Scalefactor(i)
    Sampler.sample(dS,lr0=0.0001,L=50)
    if i%500==0:
        print('step:',i,'train loss:',Sampler.y.cpu().detach().numpy(),'val loss:',Sampler.x.cpu().detach().numpy())
    if i%5000==0:
        torch.save(Sampler.net.state_dict(),'./net{}.pth'.format(i))
        np.save('./entropymap{}.npy'.format(i),Sampler.entropymap.entropy.cpu().detach().numpy())
