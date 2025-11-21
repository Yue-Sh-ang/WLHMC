import torch
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


class EntropyMap():
    '''Entropy map class, store the biased entropy map and provide methods to get and update entropy and get its gradient
    Attributes
    ----------
    entropy: 2D tensor, entropy values on the grid
    xnum,ynum: number of bins in x and y direction
    x_min,x_max,y_min,y_max: range of the entropy map
    x_width,y_width: width of each bin in x and y direction
    Methods
    -------
    get_entropy(x,y): get entropy value at (x,y)
    update_entropy(x,y,dS): update entropy value at (x,y) by dS
    get_entropy_grad(x,y,Bound,Push): get entropy gradient at (x,y)
        Bound: if True, smooth very big gradient
        Push: if True, if gradient is zero, push it to a random direction
    plot(vmin,vmax): plot the entropy map
    save(filename): save the entropy map to a .npz file
    load(filename): load the entropy map from a .npz file  
    '''
    # a linear entropy picture
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
        # Use a 6-point stencil for gradient calculation

        if x<self.x_min or x>self.x_max or y<self.y_min or y>self.y_max:
            return 0,0
        x_bin=int((x-self.x_min)/self.x_width)
        y_bin=int((y-self.y_min)/self.y_width)
        
        
        def safe_get(i, j): # Make sure indices are within bounds to avoid IndexError
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
            dSdy = 100/(1+torch.exp(-dSdy/25))-50
        if Push: #if there is no gradient, push it to a random direction
            if dSdx==0:
                dSdx=random.uniform(-5, 5)
            if dSdy==0:
                dSdy=random.uniform(-5, 5)

        return dSdx,dSdy
    
    def plot(self,vmin=None,vmax=None):
        S_masked = np.ma.masked_where(self.entropy <= 0, self.entropy) #only show positive entropy
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(S_masked, origin='lower', extent=(self.x_min, self.x_max, self.y_min, self.y_max), aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label='Entropy')
        ax.set_title('Entropy Map')
        return fig,ax
    
    def save(self, filename):
        """Save all class attributes to an .npz file."""
        np.savez(
            filename,
            entropy=self.entropy,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max
            )
    @classmethod
    def load(cls, filename):
        """Load class attributes from an .npz file and create an instance."""
        data = np.load(filename)
        instance = cls(
            Sinit=data['entropy'],
            x_min=data['x_min'].item(),
            x_max=data['x_max'].item(),
            y_min=data['y_min'].item(),
            y_max=data['y_max'].item()
        )
        return instance
        

#this shold be add: initial the entropy gradient and smooth very big gradient
class Sampler_WLHMC():
    '''Wang-Landau Hamiltonian Monte Carlo Sampler, given current state, generate a new state

    Initialization
	----------
    net: neural network to be sampled
    entropymap: biased entropy map
    fx,fy: functions to calculate x values and y values for entropy map
    datax,datay: data to be used in fx,fy
    
    Methods
    -------
    sample(dS,L,lr0,digit): perform one WLHMC sampling step, and update entropy map
        dS: entropy update step
        L: number of MD steps
        lr0: base learning rate for MD steps
        digit: if not None, snap parameters to this digit after MD steps
    return: True if accepted, False if rejected
    
    Attributes
    ----------
    net: current neural network
    entropymap: current entropy map

    '''
    def __init__(self, net, entropymap,fx,fy,datax,datay,device,boundary=None):
        self.device=device
        self.net = net 
        self.net0 = copy.deepcopy(net.state_dict()) # if reject, return to net0
        self.boundary = boundary
        self.fx=fx
        self.fy=fy
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
        xval = self.fx(self.net,self.datax)
        yval = self.fy(self.net,self.datay)
        
        X_grad, Y_grad = entropymap.get_entropy_grad(xval,yval)
        self.net.zero_grad()
        xval.backward()
        with torch.no_grad():
            E1_q_tensor= torch.cat([param.grad.view(-1) for param in self.net.parameters() if param.grad is not None])
        self.net.zero_grad()
        yval.backward()
        with torch.no_grad():
            E2_q_tensor=torch.cat([param.grad.view(-1) for param in self.net.parameters() if param.grad is not None])

        acc_tensor = -X_grad*E1_q_tensor
        acc_tensor += -Y_grad*E2_q_tensor
        return acc_tensor, xval, yval

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

    def sample(self,dS,L=20,lr0=1e-4,digit=None):
        pi=self._generate_RandomMomentum()
        self.p=pi
        self.Hi=self._cal_H()
        for _ in range(L):
            lr = lr0 * random.uniform(0.8, 1.2)
            self._move_MD(lr)
        #Hf_origin = self._cal_H()
        if digit is not None: 
            step = 10 ** (-digit)

            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    snapped = torch.round(param / step) * step
                    param.copy_(snapped)
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

        