# Install pytorch library first from documentation online
# You can also use Google Colab (uses Jupyter Notebook) for this assignment
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

############################### CUDA SETTINGS ##################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

###############################HYPER-PARAMETERS ################################
# Change these as asked in assignment
n_epochs = 30 # Number of epochs
batch_size_train = 16# Batch size for training #16
batch_size_test = 256 # Batch size for testing #256
learning_rate = 0.01 # Optimizier hyper-parameter
momentum = 0.5

###############################SETUP DATA ######################################
transform = transforms.Compose(
    [torchvision.transforms.Resize(32),
     transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# Change dataset to CIFAR here
# dataset is directory where CIFAR is stored
# You would need to download the dataset first. Please look at PyTorch
# documentation for that
train_data = datasets.CIFAR100( 'data',train=True, download=True, transform=transform)
test_data = datasets.CIFAR100( 'data',train=False, download=True, transform=transform)

# Note: DON't subset for assignment, i.e, comment or remove next 4 lines
# of code
train_idx = torch.arange(10000)
test_idx = torch.arange(500)
train_data = data_utils.Subset(train_data, train_idx)
test_data = data_utils.Subset(train_data, test_idx)

print('Train size: {}'.format(len(train_data)))
print('Test size: {}'.format(len(test_data)))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)
#Pro-tip: Try to visualize if images and labels are correctly loaded
# before training your network. Use matplotlib. Write your own code here.

def imshow(img):
    img = img/ 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale','aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'bottles', 'bowls', 'cans', 'cups', 'plates','apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers','clock', 'computer keyboard', 'lamp', 'telephone', 'television',
           'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach','bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper','cloud', 'forest', 'mountain', 'plain', 'sea', 	'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
           'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 	'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man' , 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle','hamster', 'mouse', 'rabbit', 'shrew', 'squirrel','maple', 'oak', 'palm', 'pine', 'willow',
           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train','lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',)

dataiter = iter(train_loader)

images, labels = dataiter.next()
print(labels)
# show the images
imshow(torchvision.utils.make_grid(images))
# print the labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size_train)))


###############################SETUP MODEL #####################################
class Net(nn.Module):
    def __init__(self):
        super().__init__() # 32x32
        self.conv1 = nn.Conv2d(3, 6, 5) # 32x32 -> 28x28
        self.pool = nn.MaxPool2d(2, 2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, 5) #14x14 -> 10x10
        self.pool = nn.MaxPool2d(2,2) # 10x10 -> 5x5
        self.conv3 = nn.Conv2d(16,25,5) #5x5 -> 1x1
        self.fc1 = nn.Linear(25 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100) # change to 100

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("Shape: ",x.shape)
        x = self.pool(x)
        #print("Shape: ", x.shape)
        x = F.relu(self.conv2(x))
       # print("Shape: ", x.shape)
        x = self.pool(x)
        #print("Shape: ", x.shape)
        x = F.relu(self.conv3(x))
        #print("Shape: ", x.shape)
        x = x.view(-1, 25 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device) # run on GPU
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

#################################### TRAIN #####################################

# loop over the dataset multiple times
for epoch in range(n_epochs):
    print('Epoch: {0}'.format(epoch))
    running_loss = 0.0
    # loop over mini-batch
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()

# Save weights
print('Finished Training')
# change it to something like cifar.pth
PATH = './cifar.pth'
torch.save(net.state_dict(), PATH)

##################################### TEST #####################################

#print test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
#print test images
imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size_test)))
# Reinitialize model and load weights
net = Net().to(device)
net.load_state_dict(torch.load(PATH))

outputs = net(images)
# predict output from what ground truth is given
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batch_size_test)))
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Correct Predictions: {0}'.format(correct))
print('Total Predictions: {0}'.format(total))
print('Accuracy of the network on the 500 test images: %d %%' % (100 * correct / total))
