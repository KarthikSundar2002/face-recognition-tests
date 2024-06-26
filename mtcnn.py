import torch
from torch import nn
from torchinfo import summary
import numpy as np
import os

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.prelu1 = nn.PReLU(10)
        self.max1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.prelu2 = nn.PReLU(16)
        self.conv3  = nn.Conv2d(16, 32, 3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        # a is the face classifcation output
        # b will be our bounding box regression output
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax(a)
        b = self.conv4_2(x)
        return b, a
    
    def load_weights(self, path):  
        weights = torch.load(path)
        self.load_state_dict(weights)

class RNet(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, 3)
        self.prelu1 = nn.PReLU(28)
        self.max1 = nn.MaxPool2d(3,2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, 3)
        self.prelu2 = nn.PReLU(48)
        self.max2 = nn.MaxPool2d(3,2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, 2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)
    
    def forward(self, x):
        # a is the face classifcation output
        # b will be our bounding box regression output
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        p = x.view(x.size(0), -1)
        x = self.dense4(p)
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax(a)
        b = self.dense5_2(x)
        return b, a
    
    def load_weights(self, path):  
        weights = torch.load(path)
        self.load_state_dict(weights)

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.prelu1 = nn.PReLU(32)
        self.max1 = nn.MaxPool2d(3,2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.prelu2 = nn.PReLU(64)
        self.max2 = nn.MaxPool2d(3,2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.prelu3 = nn.PReLU(64)
        self.max3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Only in O-Net we are taking the 3rd output which is the facial landmarks
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        p = x.view(x.size(0), -1)
        x = self.dense5(p)
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a
    
    def load_weights(self, path):  
        weights = torch.load(path)
        self.load_state_dict(weights)

class MTCNN(nn.Module):
    def __init__(self, image_size = 160, min_face_size=20, thresholds=[0.6,0.7,0.7], device="cpu"):
        super().__init__()

        self.image_size = image_size
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.device = torch.device(device)

        self.pNet = PNet()
        self.rNet = RNet()
        self.oNet = ONet()
        
    
    def forward(self, img, save_path=None, return_probs=False):
        b, a = self.pNet(x)
        b, a = self.rNet(x)
        b, c, a = self.oNet(x)
        return b, c, a
    
    def detect_face(self,imgs, factor):
        # Get Image and then make into the correct ordering which is (batch, channel, height, width)
        # Assumption for batch processing => Make sure that all the images are of same dimensions
        
        # following condition handles single input image
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

        type_of_model_params = next(self.pNet.parameters()).dtype
        imgs = imgs.permute(0, 3, 1, 2).dtype(type_of_model_params)

        batch_size = imgs.size(0)
        h, w = imgs.size(2), imgs.size(3)  

        pass

    def make_scale_pyramid(self, img, min_face_size):
        # we want our smallest image to be of size 12x12 as that is the input to PNet
        pass


    def load_weights(self, path):  
        weights = torch.load(path)
        self.load_state_dict(weights

if __name__ == '__main__':
    pnet_weights = np.load('./weights/pnet.npy', allow_pickle=True)
    pNet = ONet()
    pNet.load_weights('./weights/onet.pt')
    summary(pNet, input_size=(1, 3, 48, 48))