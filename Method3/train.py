import torch
import os
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.utils as utils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import io, transform


from network import *

LR = 1e-3
epochs = 10
device = 'cuda'



ALPHA = 1e13 
BETA  = 1e10  
GAMMA = 3e-2 
LAMBDA_O = 1e6
LAMBDA_F = 1e4
IMG_SIZE = (640, 360)
VGG16_MEAN = [0.485, 0.456, 0.406]
VGG16_STD = [0.229, 0.224, 0.225]

def normalizeVGG16(img):
    mean = img.new_tensor(VGG16_MEAN).view(-1, 1, 1)
    std = img.new_tensor(VGG16_STD).view(-1, 1, 1)

    img = img.div_(255.0)
    return (img - mean) / std

normalize = transforms.Lambda(lambda x: normalizeVGG16(x))

transform2 = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (1-x).mul(255)),
                normalize
                ])


def gram_matrix(input):
    # print(input.size())
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def warp(x, flo):
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def gram_matrix(input):
    temp = input.size()

    a = temp[0] 
    b = temp[1] 
    c = temp[2]
    d = temp[3]
    f = input.view(a * b, c * d)
    G = torch.mm(f, f.t()) 
    return G.div(a * b * c * d)

def toString4(num):
	string = str(num)
	while(len(string) < 4):
		string = "0"+string
	return string

class MPIDataset(Dataset):

	def __init__(self, path, transform=None):
		self.path = path+"training/"
		self.dirlist = os.listdir(self.path+"clean/")
		self.dirlist.sort()
		self.numlist = []
		for folder in self.dirlist:
			self.numlist.append(len(os.listdir(self.path+"clean/"+folder+"/")))

		self.length=sum(self.numlist)
		self.length = self.length - len(self.numlist)

#predefined functions of the parent class Dataset
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		"""
		idx must be between 0 to len-1
		flow[0] contains flow in x direction and flow[1] contains flow in y
		"""
		for i in range(0, len(self.numlist)):
			folder = self.dirlist[i]
			cleanpath = self.path+"clean/"+folder+"/"
			occpath = self.path+"occlusions/"+folder+"/"
			flowpath = self.path+"flow/"+folder+"/"
			if(idx < (self.numlist[i]-1)):
				num1, num2 = toString4(idx+1), toString4(idx+2)
				img1, img2 = io.imread(cleanpath+"frame_"+num1+".png"), io.imread(cleanpath+"frame_"+num2+".png")
				mask = io.imread(occpath+"frame_"+num1+".png")


				img1 = torch.from_numpy(transform.resize(img1, (360, 640))).to(device).permute(2, 0, 1).float()
				img2 = torch.from_numpy(transform.resize(img2, (360, 640))).to(device).permute(2, 0, 1).float()
				mask = torch.from_numpy(transform.resize(mask, (360, 640))).to(device).float()

				flow = readFlow(flowpath+"frame_"+num1+".flo")
				originalflow=torch.from_numpy(flow)                
				flow = torch.from_numpy(transform.resize(flow, (360, 640))).to(device).permute(2,0,1).float()
				flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
				flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
				# print(flow.shape) #y,x,2
				# print(img1.shape)
				break

			idx -= self.numlist[i]-1

#IMG2 should be at t in IMG1 is at T-1
		return (img1, img2, flow, mask)





addr_MPI = "/home/rajneesh/Desktop/project/MPI-Sintel-complete/" 
dataloader = DataLoader(MPIDataset(addr_MPI), batch_size=1)
model = ReCoNet().to(device)

params = model.parameters()

# lr is learning rate and adam optimizer requires parameters of the model

# loss functions
L2distance = nn.MSELoss().to(device)

adam = optim.Adam(params, lr=LR)

L2distancematrix = nn.MSELoss(reduction='none').to(device)
Vgg16 = Vgg16().to(device)

style_names = ('autoportrait', 'candy', 'composition', 'edtaonisl', 'udnie', 'starrynight')
style_img_path = './models/style/'+style_names[0]
STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]

tf = transforms.Compose([transforms.Resize((640,360)),transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255)),normalize])

temp = Image.open(style_img_path+'.jpg')
style = tf(temp)
# print(style.size())
style = style.unsqueeze(0).expand(1, 3, 640, 360).to(device)
# print(style)
# print(style.size())

for param in Vgg16.parameters():
	param.requires_grad = False

styled_featuresR = Vgg16(normalize(style))
# print(styled_featuresR.size())
# print(styled_featuresR[1].size())
# print("styled_featuresR : ",styled_featuresR.shape())

style_GM = []
for f in styled_featuresR:
	style_GM.append(gram_matrix(f))
# print(len(style_GM))

for epoch in range(epochs):
	for itr, (img1, img2, flow, mask) in enumerate(dataloader):

		flow=-flow
		print(flow)
		adam.zero_grad()
		# print(img1.size())
		
		if (itr+1) % 500 == 0:
			for param in adam.param_groups:
				temp = max(param['lr']/1.2, 1e-4)
				param['lr'] = temp
	

		feature_map1, styled_img1 = model(img1)
		# print("feature_map1 : ", feature_map1.size(), "styled_img1 : ", styled_img1.size())
		feature_map2, styled_img2 = model(img2)
		# print("feature_map2 : ", feature_map2.size(), "styled_img2 : ", styled_img2.size())
		styled_img1 = normalize(styled_img1)
		styled_img2 = normalize(styled_img2)
		# print("After normalize : styled_img1 = ", styled_img1.size(), " styled_img2  : ", styled_img2.size())
		img1 = normalize(img1) 
		img2 = normalize(img2)
		# print("After normalize : img1 = ", img1.size(), " img2  : ", img2.size())
		styled_features1, styled_features2 = Vgg16(styled_img1), Vgg16(styled_img2)

		# print("styled_features1 = ", styled_features1.size(), " styled_features2  : ", styled_features2.size())

		img_features1, img_features2 = Vgg16(img1), Vgg16(img2)
		# print("img_features1 = ", img_features1, " img_features2  : ", img_features2)

		feature_flow = nn.functional.interpolate(flow, size=feature_map1.shape[2:], mode='bilinear')
		# print("feature flow : ", feature_flow.size())
		temp = float(feature_map1.shape[2])
		feature_flow[0,0, :, :] *= temp/flow.shape[2]

		temp = float(feature_map1.shape[3])
		feature_flow[0,1, :, :] *= temp/flow.shape[3]

		feature_mask = nn.functional.interpolate(mask.view(1,1,640,360), size=feature_map1.shape[2:], mode='bilinear')

		f_temporal_loss = torch.sum(feature_mask*(L2distancematrix(feature_map2, warp(feature_map1, feature_flow))))
		f_temporal_loss = f_temporal_loss * LAMBDA_F
		f_temporal_loss = f_temporal_loss * (1/(feature_map2.shape[1]*feature_map2.shape[2]*feature_map2.shape[3]))

		# # print(styled_img1.size(), flow.size())

		# print(img2.size())
		output_term = styled_img2[0] - warp(styled_img1, flow)[0]
		# print(output_term.shape, styled_img2.shape, warped_style.shape)
		temp = img2[0] - warp(img1, flow)[0]
		# Changed the next few lines since dimension is 4 instead of 3 with batch size=1
		temp = 0.2126 * temp[0, :, :] + 0.7152 * temp[1, :, :] + 0.0722 * temp[2, :, :]
		input_term = temp.expand(output_term.size())

		s = torch.sum(mask * (L2distancematrix(output_term, input_term)))
		o_temporal_loss = s * LAMBDA_O * (1/(img1.shape[2]*img1.shape[3]))

		l2d1 = L2distance(styled_features1[2], img_features1[2].expand(styled_features1[2].size()))
		l2d2 = L2distance(styled_features2[2], img_features2[2].expand(styled_features2[2].size())) 


		content_loss = l2d1 + l2d2
		content_loss *= ALPHA/(styled_features1[2].shape[1] * styled_features1[2].shape[2] * styled_features1[2].shape[3])

		style_loss = 0
		for i, weight in enumerate(STYLE_WEIGHTS):
			gi1 = gram_matrix(styled_features1[i])
			gi2 = gram_matrix(styled_features2[i])

			l2d1 = L2distance(gi1, style_GM[i].expand(gi1.size()))
			l2d2 = L2distance(gi2, style_GM[i].expand(gi2.size()))
			style_loss += float(weight) * (l2d1 + l2d2)
		style_loss *= BETA

		# reg_loss = GAMMA *(torch.sum(torch.abs(styled_img1[:, :, :, :-1] - styled_img1[:, :, :, 1:])) + torch.sum(torch.abs(styled_img1[:, :, :-1, :] - styled_img1[:, :, 1:, :])) + torch.sum(torch.abs(styled_img2[:, :, :, :-1] - styled_img2[:, :, :, 1:])) + torch.sum(torch.abs(styled_img2[:, :, :-1, :] - styled_img2[:, :, 1:, :])))


		mid_loss = f_temporal_loss + o_temporal_loss
		# print(mid_loss)

		end_loss = style_loss + content_loss #+ reg_loss
		# print(end_loss)

		loss = mid_loss + end_loss

		loss.backward()
		adam.step()

		if (itr+1)%1000 ==0 :
			torch.save(model.state_dict(), '%s/final_reconet_epoch_%d_itr_%d.pth' % ("runs/output", epoch, itr//1000))

		print("epoch : ", epoch, " iter : ", itr)
	torch.save(model.state_dict(), '%s/reconet_epoch_%d.pth' % ("runs/output", epoch))
	print("epoch : ", epoch)
print("complete!!!!")
