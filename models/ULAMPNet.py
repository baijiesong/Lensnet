import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.transform import resize


class initial_inversion2(nn.Module):
    def __init__(self):
        super(initial_inversion2, self).__init__()

    def forward(self,meas,WL,WR):
        x0=F.leaky_relu(torch.matmul(torch.matmul(meas[:,0,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x1=F.leaky_relu(torch.matmul(torch.matmul(meas[:,1,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x2=F.leaky_relu(torch.matmul(torch.matmul(meas[:,2,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X_init=torch.cat((x0,x1,x2),3)
        X_init = X_init.permute(0,3,1,2)
        return X_init
# the gradient iteration block
class initial_inversion(nn.Module):
	def __init__(self):
		super(initial_inversion,self).__init__()
	#	self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

	def forward(self,Xinp,Z,phil,phir):
		y0=F.leaky_relu(torch.matmul(torch.matmul(Z[:,0,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
		y10=F.leaky_relu(torch.matmul(torch.matmul(Z[:,1,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
		y11=F.leaky_relu(torch.matmul(torch.matmul(Z[:,2,:,:],phir[:,:,0]).permute(0,2,1),phil[:,:,0]).permute(0,2,1).unsqueeze(3))
	#	y2=F.leaky_relu(torch.matmul(torch.matmul(Z[:,3,:,:],phir[:,:,3]).permute(0,2,1),phil[:,:,3]).permute(0,2,1).unsqueeze(3))
		Y_init=torch.cat((y0,y10,y11),3)
		y_init = Y_init.permute(0,3,1,2)    
		R = y_init + Xinp  
		return R
      
 #Attention    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes , 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
      
                  
# Define IAMP-Net Block
class BasicBlock(torch.nn.Module):
	def __init__(self,phil,phir):
		super(BasicBlock, self).__init__()
		self.soft_thr = nn.Parameter(torch.Tensor([0.0005]))
		self.tau = nn.Parameter(torch.Tensor([0.25]))
		self.PhiL = nn.Parameter(torch.tensor(phil),requires_grad=False)
		self.PhiR = nn.Parameter(torch.tensor(phir),requires_grad=False)
		self.eta = nn.Parameter(torch.Tensor([2]))
		self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 3, 3, 3)))
		self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))		
		self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
		self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))		
		self.conv4_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(3, 32, 3, 3)))
		self.gradLayer = initial_inversion()
		self.ini_inversion = initial_inversion2()
	def forward(self, meas, x, Z):
		R = self.gradLayer(x,Z,self.PhiL,self.PhiR)
		x = F.conv2d(R, self.conv1_forward, padding=1)
		R1 = F.relu(x)
		x = F.conv2d(R1, self.conv2_forward, padding=1)
		x = x+R1
		R2 = F.relu(x)
		x = F.conv2d(R2, self.conv3_forward, padding=1)
		x = x+R2
		x = F.relu(x)
		x_forward = F.conv2d(x, self.conv4_forward, padding=1)
		#x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
		mu1 = torch.sign(F.relu(torch.mul(torch.abs(x_forward)-self.soft_thr,self.soft_thr*self.eta-torch.abs(x_forward))))
		mu2 = torch.sign(F.relu(torch.abs(x_forward)-self.soft_thr*self.eta))

		L = torch.mul(mu1,torch.mul(self.eta/(self.eta-1),torch.mul(torch.abs(x_forward)-self.soft_thr,torch.sign(x_forward))))+torch.mul(mu2,x_forward)
		x = F.conv2d(L, self.conv1_backward, padding=1)
		L1 = F.relu(x)
		x = F.conv2d(L1, self.conv2_backward, padding=1)
		x = x+L1
		L2 = F.relu(x)
		x = F.conv2d(L2, self.conv3_backward, padding=1)
		x = x+L2
		x = F.relu(x)
		x_backward = F.conv2d(x, self.conv4_backward, padding=1)

		x = F.conv2d(x_forward, self.conv1_backward, padding=1)
		x = x+L
		L1 = F.relu(x)
		x = F.conv2d(L1, self.conv2_backward, padding=1)
		x = x+L1
		L2 = F.relu(x)
		x = F.conv2d(L2, self.conv3_backward, padding=1)
		x = x+L2
		x = F.relu(x)
		x_est = F.conv2d(x, self.conv4_backward, padding=1)

		Z = meas- self.ini_inversion(x_backward,self.PhiL.permute(1,0,2),self.PhiR.permute(1,0,2))+torch.mul(self.tau,Z)

		symloss = x_est - R

		return [x_backward, Z, symloss]

# Define IAMP-Net
class ULAMPNet(torch.nn.Module):
	def __init__(self, LayerNo=7):
		super(ULAMPNet, self).__init__()
		onelayer = []
		# phil = np.random.rand(320, 320, 3).astype(np.float32)
		# phir = np.random.rand(320, 320, 3).astype(np.float32)
		# WL = np.random.rand(320, 320, 1).astype(np.float32)
		# WR = np.random.rand(320, 320, 1).astype(np.float32)
		# phil = np.random.rand(270, 270, 1).astype(np.float32)
		# phir = np.random.rand(480, 480, 1).astype(np.float32)
		# WL = np.random.rand(270, 270, 1).astype(np.float32)
		# WR = np.random.rand(480, 480, 1).astype(np.float32)
		MWDNS = False
		if MWDNS:
			psf = Image.open('MWDNs_psf.png').convert('L')
			psf = np.array(psf)
			psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
			psf /= np.linalg.norm(psf.ravel())
			psf = np.expand_dims(psf, axis=2)
		else:
			psf = Image.open('Mirflickr_psf.tiff').convert('L')
			psf = np.array(psf)
			psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
			ds = 4
			psf = resize(
				psf, (psf.shape[0]//ds, psf.shape[1]//ds), mode='constant', anti_aliasing=True)
			psf /= np.linalg.norm(psf.ravel())
			psf = torch.from_numpy(psf).unsqueeze(2)
		self.LayerNo = LayerNo
		if MWDNS:
			self.PhiL = nn.Parameter(torch.tensor(psf))
			self.PhiR = nn.Parameter(torch.tensor(psf))
			self.WL = nn.Parameter(torch.tensor(psf))
			self.WR = nn.Parameter(torch.tensor(psf))
			phil = psf
			phir = psf
		else:
			# torch.Size([270, 270, 1]) torch.Size([480, 480, 1]) torch.Size([270, 270, 1]) torch.Size([480, 480, 1])
			psf_t1 = psf.permute(2, 0, 1)
			psf_t2 = psf.permute(2, 1, 0)
			self.PhiL = nn.Parameter(
				torch.bmm(psf_t1, psf_t2).permute(1, 2, 0))
			self.PhiR = nn.Parameter(
				torch.bmm(psf_t2, psf_t1).permute(1, 2, 0))
			self.WL = nn.Parameter(torch.bmm(psf_t1, psf_t2).permute(1, 2, 0))
			self.WR = nn.Parameter(torch.bmm(psf_t2, psf_t1).permute(1, 2, 0))
			phil = self.PhiL.cpu().detach().numpy()
			phir = self.PhiR.cpu().detach().numpy()
		for i in range(LayerNo):
			onelayer.append(BasicBlock(phil, phir))

		self.fcs = nn.ModuleList(onelayer)
		self.ini_inversion = initial_inversion2()
#		self.enhancement = DUnet(4,3,32) 
	def forward(self, meas):
		x = self.ini_inversion(meas,self.WL,self.WR)


		Y = self.ini_inversion(x,self.PhiL.permute(1,0,2),self.PhiR.permute(1,0,2))

		Z = meas - Y		
		x_init = x
		layers_sym = []   # for computing symmetric loss
		for i in range(self.LayerNo):
			[x, Z, layer_sym] = self.fcs[i](meas, x, Z)
			layers_sym.append(layer_sym)			
#		x_final = self.enhancement(x_init,x)
		x_final = x
		layers_sym.append(x_final)
		return layers_sym


if __name__ == '__main__':
    t = torch.rand(1, 3, 270, 480).cuda()
    model = ULAMPNet().cuda()
    model.eval()
    with torch.no_grad():
        res = model(t)
        for i in range(len(res)):
            print(res[i].shape)
