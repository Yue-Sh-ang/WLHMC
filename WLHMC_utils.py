import torch
import random
import copy
import numpy as np


class EntropyMap():
    '''Entropy map class, store the biased entropy map and provide methods to get and update entropy and get its gradient'''
    # a logspace entropy picture
    def __init__(self,Sinit,x_min,x_max,y_min,y_max):
        self.entropy=Sinit
        self.xnum=Sinit.shape[0]
        self.ynum=Sinit.shape[1]
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.x_width=(x_max-x_min)/self.xnum
        self.y_width=(y_max-y_min)/self.ynum

    def get_entropy(self,x,y):
        if x<self.x_min or x>self.x_max or y<self.y_min or y>self.y_max:
            return 0
        x_bin=int((x-self.x_min)/self.x_width)
        y_bin=int((y-self.y_min)/self.y_width)
        return self.entropy[x_bin,y_bin]
    
    def update_entropy(self,x,y,dS):
        if x<self.x_min or x>self.x_max or y<self.y_min or y>self.y_max:
            return
        x_bin=int((x-self.x_min)/self.x_width)
        y_bin=int((y-self.y_min)/self.y_width)
        self.entropy[x_bin,y_bin]+=dS

    def get_entropy_grad(self,x,y,Bound=True,Push=True):
        if x<self.x_min or x>self.x_max or y<self.y_min or y>self.y_max:
            return 0,0
        x_bin=int((x-self.x_min)/self.x_width)
        y_bin=int((y-self.y_min)/self.y_width)
        
        # Use a 6-point stencil for gradient calculation
        def safe_get(i, j):
        # Make sure indices are within bounds to avoid IndexError
            if 0 <= i < self.xnum and 0 <= j < self.ynum:
                return self.entropy[i, j]
            else:
                return 0.0

        dSdx = 0.0
        for offset in [1, 0, -1, -2]:
            dSdx += (
            -3 * safe_get(x_bin - 3, y_bin + offset)
            -5 * safe_get(x_bin - 2, y_bin + offset)
            -15 * safe_get(x_bin - 1, y_bin + offset)
            +15 * safe_get(x_bin, y_bin + offset)
            +5 * safe_get(x_bin + 1, y_bin + offset)
            +3 * safe_get(x_bin + 2, y_bin + offset)
            ) / self.x_width / 180

        dSdy = 0.0
        for offset in [1, 0, -1, -2]:
            dSdy += (
            -3 * safe_get(x_bin + offset, y_bin - 3)
            -5 * safe_get(x_bin + offset, y_bin - 2)
            -15 * safe_get(x_bin + offset, y_bin - 1)
            +15 * safe_get(x_bin + offset, y_bin)
            +5 * safe_get(x_bin + offset, y_bin + 1)
            +3 * safe_get(x_bin + offset, y_bin + 2)
            ) / self.y_width / 180
        
        if Bound: #smooth very big gradient
            dSdx = 100/(1+torch.exp(-dSdx/25))-50
            dSdy = 100/(1 + torch.exp(-dSdy/25))-50
        if Push: #if there is no gradient, push it to a random direction
            if dSdx==0:
                dSdx=random.uniform(-5, 5)
            if dSdy==0:
                dSdy=random.uniform(-5, 5)

        return dSdx,dSdy

#this shold be add: initial the entropy gradient and smooth very big gradient
class Sampler_WLHMC():
    '''Wang-Landau Hamiltonian Monte Carlo Sampler, given current state, generate a new state

    Initialization
	----------
    net: neural network to be sampled
    entropymap: biased entropy map
    cal_loss: loss function
    datax,datay: data for E1 and E2
    
    Methods
    -------
    sample(dS,L,lr0): perform one WLHMC sampling step, and update entropy map
        dS: entropy update step
        L: number of MD steps
        lr0: base learning rate for MD steps
    return: True if accepted, False if rejected
    
    Attributes
    ----------
    net: current neural network
    entropymap: current entropy map

    '''
    def __init__(self, net, entropymap,cal_loss,datax,datay,device,boundary=None):
        self.device=device
        self.net = net 
        self.net0 = copy.deepcopy(net.state_dict()) # if reject, return to net0
        self.boundary = boundary
        self.cal_loss=cal_loss
        self.datax=datax
        self.datay=datay

        self.entropymap=entropymap
        # initial location,momentum,acceleration and Hamiltonian
        self.acc,self.x,self.y=self._cal_acceleration()
        pi=self._generate_RandomMomentum()
        self.p=pi
        self.Hi=self._cal_H()
        
   
    def _generate_RandomMomentum(self):
        # Mass matrix=delta_ij; sqrt(sum(p^2))~sqrt(sum(q^2))~weight
        momentum = []
        num_para = 0
        for param in self.net.parameters():
            length = param.data.numel()
            momentum.append(torch.normal(0, 2/np.sqrt(length), param.data.shape,device=self.device))
            num_para += param.data.numel()

        return  momentum
    
    def _cal_acceleration(self):
        entropymap=self.entropymap
        loss_train = self.cal_loss(self.net,self.datax)
        ln_loss_train =torch.log(loss_train)
        loss_val = self.cal_loss(self.net,self.datay)
        ln_loss_val=torch.log(loss_val)
        X_grad, Y_grad = entropymap.get_entropy_grad(ln_loss_train,ln_loss_val)
        self.net.zero_grad()
        ln_loss_train.backward()
        with torch.no_grad():
            E1_q_tensor= torch.cat([param.grad.view(-1) for param in self.net.parameters() if param.grad is not None])
        self.net.zero_grad()
        ln_loss_val.backward()
        with torch.no_grad():
            E2_q_tensor=torch.cat([param.grad.view(-1) for param in self.net.parameters() if param.grad is not None])

        acc_tensor = -X_grad*E1_q_tensor
        acc_tensor += -Y_grad*E2_q_tensor
        return acc_tensor,ln_loss_train,ln_loss_val

    def _cal_H(self):
        K_total = []
        for vel in self.p:
            K_total.append((vel**2).sum())
        sumKtotal=0.5*sum(K_total)
        H=sumKtotal+self.entropymap.get_entropy(self.x,self.y)
        return H.cpu().detach().numpy()

    def _check_boundary(self,param_tensor,velocity_tensor,boundary):
        with torch.no_grad():
            # Reflecting boundary conditions
            lower_mask = param_tensor < -boundary
            upper_mask = param_tensor > boundary

            # Reflect upper boundary
            param_tensor[upper_mask] = boundary[upper_mask] - ( param_tensor[upper_mask]-boundary[upper_mask] )
            velocity_tensor[upper_mask] = -velocity_tensor[upper_mask]

            # Reflect lower boundary
            param_tensor[lower_mask] = -boundary[lower_mask] + (-boundary[lower_mask]-param_tensor[lower_mask] )
            velocity_tensor[lower_mask] = -velocity_tensor[lower_mask]

        return param_tensor, velocity_tensor

    def _move_MD(self,lr):
 
        with torch.no_grad():
            param_tensor = torch.cat([param.view(-1) for param in self.net.parameters()]) #copy parameters as a flatten tensor
            velocity_tensor = torch.cat([vel.view(-1) for vel in self.p])#copy velocities as a flatten tensor
            velocity_tensor += self.acc * lr *0.5 #Frog leap step1
            param_tensor += velocity_tensor * lr #Frog leap step2
            if self.boundary is not None:
                param_tensor,velocity_tensor=self._check_boundary(param_tensor,velocity_tensor,self.boundary) #reflecting boundary condition
            
            # update net parameters
            offset=0
            for param in self.net.parameters():
                param_length = param.numel()
                param.copy_(param_tensor[offset:offset + param_length].view(param.shape))
                offset += param_length

        self.acc,self.x,self.y=self._cal_acceleration() #update location and acceleration
        
        with torch.no_grad():
            velocity_tensor += self.acc * lr *0.5 #Frog leap step3
            offset = 0
            for i, vel in enumerate(self.p):
                vel_length = vel.numel()
                self.p[i].copy_(velocity_tensor[offset:offset + vel_length].view(vel.shape))
                offset += vel_length

    def sample(self,dS,L=10,lr0=1e-3):
        pi=self._generate_RandomMomentum()
        self.p=pi
        self.Hi=self._cal_H()
        for _ in range(L):
            lr = lr0 * random.uniform(0.8, 1.2)
            self._move_MD(lr)
        Hf = self._cal_H()
        if np.random.rand() < np.exp(self.Hi - Hf):
            # accept
            self.net0 = copy.deepcopy(self.net.state_dict())
            self.entropymap.update_entropy(self.x,self.y,dS)
            self.acc,self.x,self.y=self._cal_acceleration()
            return True
        else:
            # reject
            self.net.load_state_dict(self.net0)
            self.entropymap.update_entropy(self.x,self.y,dS)
            self.acc,self.x,self.y=self._cal_acceleration()
            return False

        