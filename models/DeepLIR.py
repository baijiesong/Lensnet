import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
    

class DeepLIR(nn.Module):
    def __init__(self):
        super(DeepLIR, self).__init__()
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
                           learning_options = learning_options_admm, cuda_device ="cpu")
        self.denoise_model = SwinIR(upscale=1, in_chans=3, img_size=(psf.shape[0], psf.shape[1]), window_size=8, img_range=1., depths=[1,1,1], embed_dim=180, num_heads=[6,6,6], mlp_ratio=2, upsampler='', resi_connection='1conv')

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
    t = torch.rand(1, 3, 270, 480)
    model = DeepLIR()
    model.eval()
    with torch.no_grad():
        out = model(t)
        print(out.shape)