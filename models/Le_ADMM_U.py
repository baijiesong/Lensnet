import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
from einops import rearrange
from inspect import isfunction
from torch import einsum
from PIL import Image
from skimage.transform import resize


def L_tf(a):
    xdiff = a[:,:, 1:, :]-a[:,:, :-1, :]
    ydiff = a[:,:, :, 1:]-a[:,:, :, :-1]
    return -xdiff, -ydiff

####### Soft Thresholding Functions  #####

def soft_2d_gradient2_rgb(model, v,h,tau):
    

    z0 = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    z1 = torch.zeros(model.batch_size, 3, 1, model.DIMS1*2, dtype = torch.float32, device=model.cuda_device)
    z2 = torch.zeros(model.batch_size, 3, model.DIMS0*2, 1, dtype= torch.float32, device=model.cuda_device)

    vv = torch.cat([v, z1] , 2)
    hh = torch.cat([h, z2] , 3)
    
    #adding some small value so to solve the non gradient 
    mag = torch.sqrt(vv*vv + hh*hh+torch.tensor(1.11e-14))
    magt = torch.max(mag - tau, z0, out=None)
    mag = torch.max(mag - tau, z0, out=None) + tau
    mmult = magt/(mag)#+1e-5)

    return v*mmult[:,:, :-1,:], h*mmult[:,:, :,:-1]

# computes the HtX
def Hadj(model,x):
    xc = torch.zeros_like(x, dtype=torch.float32)
    x_complex=torch.complex(x, xc)
    X = torch.fft.fft2(x_complex)
    Hconj=model.Hconj_new
    
    HX = Hconj*X
    out = torch.fft.ifft2(HX)
    out_r=out.real
    return out_r

#computes the uk+1
def Ltv_tf(a, b): 
    return torch.cat([a[:,:, 0:1,:], a[:,:, 1:, :]-a[:,:, :-1, :], -a[:,:,-1:,:]],
                2) + torch.cat([b[:,:,:,0:1], b[:, :, :, 1:]-b[:, :, :,  :-1], -b[:,:, :,-1:]],3)
    
#takes the real matrix and return the corresponplex matrix
def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.complex(r, i) 

#computes the Hx+1
def Hfor(model, x):
    xc = torch.zeros_like(x, dtype=torch.float32)
    x_complex=torch.complex(x, xc)
    #print(x.shape)
    X = torch.fft.fft2(x_complex)
    HX = model.H*X
    out = torch.fft.ifft2(HX)
    return out.real

######## ADMM Parameter Update #########
def param_update(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    else:
        mu_up = mu
        
    if s > res_tol*r:
        mu_up = mu_up/mu_dec
    else:
        mu_up = mu_up

def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, :, C01:C02, C11:C12]

def TVnorm_tf(x):
    x_diff, y_diff = L_tf(x)
    result = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
    return result

######## normalize image #########
def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized


def admm(model, in_vars, alpha2k_1, alpha2k_2, CtC, Cty, mu_auto, n, y):
    """
    performs one iteration of ADMM
    """    
    
    sk = in_vars[0]
    alpha1k = in_vars[1] 
    alpha3k = in_vars[2]
    Hskp = in_vars[3]; 
    
    #if the autotune is enabled mu are from the mu_auto parameter
    if model.autotune == True:
        mu1 = mu_auto[0]
        mu2 = mu_auto[1]
        mu3 = mu_auto[2]
    #else mu are n-1 values of the mu list
    else:
        mu1 = model.mu1[n]
        mu2 = model.mu2[n]
        mu3 = model.mu3[n]
    # tau is the n-1 values of the iterations 
    tau = model.tau[n] #model.mu_vals[3][n]
    
    dual_resid_s = []
    primal_resid_s = []
    dual_resid_u = []
    primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []
    #print(mu1.device)

    Smult = 1/(mu1.to(model.cuda_device)*model.HtH.to(model.cuda_device) + mu2.to(model.cuda_device)*model.LtL.to(model.cuda_device) + mu3.to(model.cuda_device))  # May need to expand dimensions 
    Vmult = 1/(CtC + mu1)
    
    ###############  update u = soft(Ψ*x + η/μ2,  tau/μ2) ###################################
    Lsk1, Lsk2 = L_tf(sk)        # X and Y Image gradients 
    ukp_1, ukp_2 = soft_2d_gradient2_rgb(model, Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau)
    
    ################  update      ######################################

    vkp = Vmult*(mu1*(alpha1k/mu1 + Hskp) + Cty)
    
    ################  update w <-- max(alpha3/mu3 + sk, 0) ######################################


    zero_cuda = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
        
    wkp = torch.max(alpha3k/mu3 + sk, zero_cuda, out=None)
   

    
    # no learned prox 
    skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj(model, vkp - alpha1k/mu1) + mu2*Ltv_tf(ukp_1 - alpha2k_1/mu2, ukp_2 - alpha2k_2/mu2) 
    symm = []
  
    #SKP_numerator = torch.fft.fft(make_complex(skp_numerator), 2)
    SKP_numerator = torch.fft.fft2(make_complex(skp_numerator))

    skp = (torch.fft.ifft2((make_complex(Smult)* SKP_numerator))).real
    
    Hskp_up = Hfor(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * torch.norm(Hskp - Hskp_up))
    primal_resid_s.append(torch.norm(r_sv))

    # Autotune
    if model.autotune == True:
        mu1_up = param_update(mu1, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_s[-1], dual_resid_s[-1])
        #model.mu_vals[0][n+1] = model.mu_vals[0][n+1] + mu1_up
    else: 
        if n == model.iterations-1:
            mu1_up = model.mu_vals[0][n]
        else:
            mu1_up = model.mu_vals[0][n+1]

    alpha1kup = alpha1k + mu1*r_sv

    Lskp1, Lskp2 = L_tf(skp)
    r_su_1 = Lskp1 - ukp_1
    r_su_2 = Lskp2 - ukp_2

    dual_resid_u.append(mu2*torch.sqrt(torch.norm(Lsk1 - Lskp1)**2 + torch.norm(Lsk2 - Lskp2)**2))
    primal_resid_u.append(torch.sqrt(torch.norm(r_su_1)**2 + torch.norm(r_su_2)**2))

    if model.autotune == True:
        mu2_up = param_update(mu2, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_u[-1], dual_resid_u[-1])
    else:
        if n == model.iterations-1:
            mu2_up = model.mu_vals[1][n]
        else:
            mu2_up = model.mu_vals[1][n+1]

    alpha2k_1up= alpha2k_1 + mu2*r_su_1
    alpha2k_2up= alpha2k_2 + mu2*r_su_2

    r_sw = skp - wkp
    dual_resid_w.append(mu3*torch.norm(sk - skp))
    primal_resid_w.append(torch.norm(r_sw))

    if model.autotune == True:
        mu3_up = param_update(mu3, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_w[-1], dual_resid_w[-1])
    else:
        if n == model.iterations-1:
            mu3_up = model.mu_vals[2][n]
        else:
            mu3_up = model.mu_vals[2][n+1]

    alpha3kup = alpha3k + mu3*r_sw

    data_loss = torch.norm(crop(model, Hskp_up)-y)**2
    tv_loss = tau*TVnorm_tf(skp)

    
    if model.printstats == True:
        
        admmstats = {'dual_res_s': dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_s':  primal_resid_s[-1].cpu().detach().numpy(),
                     'dual_res_w':dual_resid_w[-1].cpu().detach().numpy(),
                     'primal_res_w':primal_resid_w[-1].cpu().detach().numpy(),
                     'dual_res_u':dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_u':primal_resid_s[-1].cpu().detach().numpy(),
                     'data_loss':data_loss.cpu().detach().numpy(),
                     'total_loss':(data_loss+tv_loss).cpu().detach().numpy()}
        
        
        print('\r',  'iter:', n,'s:', admmstats['dual_res_s'], admmstats['primal_res_s'], 
         'u:', admmstats['dual_res_u'], admmstats['primal_res_u'],
          'w:', admmstats['dual_res_w'], admmstats['primal_res_w'], end='')
    else:
        admmstats = []

    
    #out vars contains X, alpha1k+1, alpha3k+1, Hxk+1
    out_vars = torch.stack([skp, alpha1kup, alpha3kup, Hskp_up])

    #updated value of mues
    mu_auto_up = torch.stack([mu1_up, mu2_up, mu3_up])
    # returns outvars, alpha2's, myus, admmstats
    return out_vars, alpha2k_1up, alpha2k_2up, mu_auto_up, symm, admmstats



def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

#makes the laplacian filer used as the sparsyfying tranform
def make_laplacian(model):
    lapl = np.zeros([model.DIMS0*2,model.DIMS1*2])
    lapl[0,0] =4.; 
    lapl[0,1] = -1.; lapl[1,0] = -1.; 
    lapl[0,-1] = -1.; lapl[-1,0] = -1.; 

    LTL = np.abs(np.fft.fft2(lapl))
    # LTL = np.abs(np.fft.fft2(lapl))
    return LTL


class ADMM_Net(nn.Module):
    
    def __init__(self,batch_size,h, iterations, cuda_device, learning_options={'learned_vars':[]},
     le_admm_s=False,denoise_model=[]):
        """
        constructor to initialize the ADMM network.

        Args:
            batch_size (int): Batch size
            h (np.array): PSF of the imaging system size (270,480)
            iterations (int): number of unrolled iterations
            learning_options (dict, optional): variables to be learned Defaults to {'learned_vars':[]}.
            cuda_device (str, optional): device {cuda or cpu}. Defaults to torch.device('cpu').
            le_admm_s (bool, optional): Turn on if using Le-ADMM*, otherwise should be set to False. Defaults to False.
            denoise_model (list, optional): model to use as a learnable regularizer. Defaults to [].
        """
        super(ADMM_Net, self).__init__()        
        #number of unrolled iterations
        
        self.iterations=iterations
        #batch size
        self.batch_size=batch_size
        #using autotune
        self.autotune=False
        #real data or the simulated data
        self.realdata=True
        #print ADMM variables
        self.printstats=False
        
        # add noise ( only if simulated data)
        self.addnoise=False
        # noise standard deviation
        self.noise_std=0.05
        self.cuda_device=cuda_device
        self.l_admm_s=le_admm_s
        if le_admm_s==True:
            self.Denoiser=denoise_model.to(cuda_device)
           
        #learned structure options
        self.learning_options=learning_options 
        
        #initialize constants
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions

        self.PAD_SIZE0 = int((self.DIMS0)//2)  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)  # Pad size
        #initialize variables
        self.initialize_learned_variables(learning_options)
        #psf
        self.h_var=torch.nn.Parameter(torch.tensor(h,dtype=torch.float32, device=self.cuda_device),requires_grad=False)
        self.h_padded=F.pad(self.h_var,(self.PAD_SIZE1,self.PAD_SIZE1,self.PAD_SIZE0,self.PAD_SIZE0),'constant',0)
        
        # shift the zero frequency component from center to the corner
        self.h_center_right=torch.fft.fftshift(self.h_padded)
        
        # compute the 2D discrete Fourier transform
        self.H=torch.fft.fft2(self.h_center_right)
        
        self.Hconj_new=torch.conj(self.H)
        self.HtH=self.H*self.Hconj_new
        self.HtH=self.HtH.real
        #LtL is the sparsifying transformation 
        self.LtL = torch.nn.Parameter(torch.tensor(make_laplacian(self), dtype=torch.float32, device=self.cuda_device),
                                      requires_grad=False)
        self.resid_tol =  torch.tensor(1.5, dtype= torch.float32, device=self.cuda_device)
        # change of penalizing factor
        self.mu_inc = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device)
        self.mu_dec = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device)
        
    def initialize_learned_variables(self,learning_options):
            #mu are the scaler penalty parameters  for each iterations
            if 'mus' in learning_options['learned_vars']:  
                #initialize to small value i.e 1e-04 for each iteration
                self.mu1= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
                self.mu2= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
                self.mu3= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
            else:
                #initialize to small value but doesn't make it learnable
                self.mu1=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
                self.mu2=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
                self.mu3 = torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
            
            if "tau" in learning_options['learned_vars']: # tau parameter 
                self.tau = torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*2e-4,dtype=torch.float32,device=self.cuda_device))
            
                #  initialize to small value
            else:
                self.tau=torch.ones(self.iterations, dtype = torch.float32,device=self.cuda_device)*2e-3
        
    def forward(self, inputs):  
        self.batch_size=inputs.shape[0]
        #mu and tau parameters
        self.mu_vals=torch.stack([self.mu1,self.mu2,self.mu3,self.tau])
        self.admmstats = {'dual_res_s': [], 'dual_res_u': [], 'dual_res_w': [], 
             'primal_res_s': [], 'primal_res_u': [], 'primal_res_w': [],
             'data_loss': [], 'total_loss': []}
        if self.autotune==True:
            self.mu_auto_list= {'mu1': [], 'mu2': [], 'mu3': []}
        y = inputs.to(self.cuda_device)
        Cty = pad_zeros_torch(self, y).to(self.cuda_device) #(Ctx)
        CtC = pad_zeros_torch(self, torch.ones_like(y,device=self.cuda_device))     # Zero padded ones with the shape of input y (CtC)
            
        in_vars = [] 
        in_vars1 = []
        in_vars2 = []
        Hsk_list = []
        a2k_1_list=[]
        a2k_2_list= []
        
        sk = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        # larange multipliers
        alpha1k = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        alpha3k = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        #Hxk from the paper for the vkp
        Hskp = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        
        # if learnable addam is set true( used for Le-ADMM-*)
        if self.l_admm_s == True:
            # use of U-net as a denoiser after the iteration ie. sk is feed towards the U-net
            Lsk_init, mem_init = self.Denoiser.forward(sk)
            # set aplha2k as the size of the output after the denoiser , drop the alpha2k
            alpha2k = torch.zeros_like(Lsk_init, dtype = torch.float32,  device=self.cuda_device)
        
        else:
            alpha2k_1 = torch.zeros_like(sk[:,:,:-1,:], dtype = torch.float32,device=self.cuda_device)  
            alpha2k_2 = torch.zeros_like(sk[:,:,:,:-1], dtype = torch.float32,device=self.cuda_device)
            
            a2k_1_list.append(alpha2k_1)
            a2k_2_list.append(alpha2k_2)                             
        mu_auto = torch.stack([self.mu1[0], self.mu2[0], self.mu3[0], self.tau[0]])
        in_vars.append(torch.stack([sk, alpha1k, alpha3k, Hskp]))
        
        for i in range(0,self.iterations):
            
           out_vars, a_out1, a_out2, _ , symm, admmstats = admm(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, [], i, y)
           in_vars.append(out_vars)
           a2k_1_list.append(a_out1)
           a2k_2_list.append(a_out2)
           
        x_out = crop(self, in_vars[-1][0])
        x_outn = normalize_image(x_out)
        self.in_list = in_vars
        return x_outn
    

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

#not required


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y",
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=False,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in,
                                    time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time=None):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)

            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            temp = h.pop()
            _, channels, height, width = temp.size()
            x = F.interpolate(x, size=(height, width), mode='bilinear')
            x = torch.cat((x, temp), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class Le_ADMM_U(nn.Module):
    def __init__(self):
        super(Le_ADMM_U, self).__init__()
        var_options = {'plain_admm': [],
                       'mu_and_tau': ['mus', 'tau'],
                       }
        learning_options_admm = {'learned_vars': var_options['mu_and_tau']}
        MWDNS = False
        if MWDNS:
            psf = Image.open('MWDNs_psf.png').convert('L')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            psf /= np.linalg.norm(psf.ravel())
        else:
            psf = Image.open('Mirflickr_psf.tiff').convert('L')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            ds = 4
            psf = resize(
                psf, (psf.shape[0]//ds, psf.shape[1]//ds), mode='constant', anti_aliasing=True)
            psf /= np.linalg.norm(psf.ravel())
        self.admm_model = ADMM_Net(batch_size = 1, h = psf, iterations = 5, 
                           learning_options = learning_options_admm, cuda_device ="cuda")
        self.denoise_model = Unet(dim=36,channels=3,dim_mults=(1,2,4,8))

    def forward(self, x):

        admm_output = self.admm_model(x)
        #pad input image to be a multiple of window_size (pad to the right and bottom)

        # _, _, h_old, w_old = admm_output.size()
        # window_size=8
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # img_lq = torch.cat([admm_output, torch.flip(admm_output, [2])], 2)[:, :, :h_old + h_pad, :]
        # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        # final_output = self.denoise_model(admm_output)
        # # return to the orginal shape by cropping
        # final_output=final_output[:,:,:h_old,:w_old]

        final_output = self.denoise_model(admm_output)
        return final_output

    def to(self, indevice):
        self = super().to(indevice)
        self.admm_model.to(indevice)
        self.admm_model.h_var.to(indevice)
        #self.admm_model.h_zeros.to(indevice)
        #self.admm_model.h_complex.to(indevice)
        self.admm_model.LtL.to(indevice)
        return self


if __name__ == '__main__':
    t = torch.rand(1, 3, 270, 480).cuda()
    model = Le_ADMM_U().cuda()
    out = model(t)
    print(out.shape)