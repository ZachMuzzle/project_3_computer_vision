import os
import torch
import torchvision
import torch.utils.data as data_utils
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from PIL import Image
from tqdm import tqdm
# segmap maps for all the different possible images
def segmap(image,nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for i in range(0,nc):
        idx = image == i
        r[idx] = label_colors[i,0]
        g[idx] = label_colors[i,1]
        b[idx] = label_colors[i,2]
    rgb = np.stack([r,g,b],axis=2)
    return rgb
# our model
fcn = models.segmentation.fcn_resnet50(pretrained=True).eval()
# Just prints our model
print(fcn)

############ CHANGE IMAGE HERE #####################
img = Image.open('./train.jpg')
###################################################
plt.figure(figsize=(10,5))
plt.imshow(img);plt.show()
#img = cv2.imread('bird.jpg')
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); plt.show()
# fix this. # Now fixed just removed resize and added mean and std.
transform = transforms.Compose(
    [
     transforms.ToTensor(),
     torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp = transform(img).unsqueeze(0)
print(inp.size())
###########################
# Load image in might take awhile
output = fcn(inp)['out']
#Display feature map
for feature_map in output:
    im = np.squeeze(feature_map.detach().cpu().numpy())
    im = np.transpose(im,[1,2,0])
    plt.figure(figsize=(10,10))
    for i in tqdm(range(21), desc="Loading..."):
        ax = plt.subplot(5,5,i+1)

        plt.imshow(im[:,:,i], cmap='gray')
    plt.show()
#####################
#use gpu
if torch.cuda.is_available():
    inp = inp.to("cuda:0")
    fcn.to("cuda:0")

om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
print(om)
print(om.shape)
print(np.unique(om))

#call segmap function
rgb = segmap(om)
plt.imshow(rgb);plt.show()