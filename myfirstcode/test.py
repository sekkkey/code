import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net import SqueezeNet


class SingleClassDataset(Dataset):


    def __init__(self, file_path):
        # 保证输入的是正确的路径
        if not os.path.isdir(file_path):
            raise ValueError("input file_path is not a dir")
        self.file_path = file_path
        self.file_path_high = os.path.join(self.file_path,'high/')
        self.file_path_low = os.path.join(self.file_path, 'low/')
        # 获取路径下所有的图片名称，必须保证路径内没有图片以外的数据
        self.image_list_high = os.listdir(self.file_path_high)
        self.image_list_low = os.listdir(self.file_path_low)
        # 将PIL的Image转为Tensor
        self.transforms = T.ToTensor()

    def __getitem__(self, index):
        # 根据index获取图片完整路径
        image_path_high = os.path.join(self.file_path_high, self.image_list_high[index])
        # 都图片并转为Tensor
        image_high = self._read_convert_image(image_path_high)
        image_path_low = os.path.join(self.file_path_low, self.image_list_low[index])
        # 都图片并转为Tensor
        image_low = self._read_convert_image(image_path_low)
        return (image_high,image_low)

    def _read_convert_image(self, image_name):
        image = Image.open(image_name)
        #image = self.transforms(image).float()

        tf_resize = transforms.Compose([transforms.Resize((224,224))])
        image = tf_resize(image)
        image = T.ToTensor()(image)
        #image = torch.reshape(image,(-1,3,224,224))
        return image

    def __len__(self):
        return len(self.image_list_high)

model_path = 'log/model_.pkl'
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device1 = torch.device('cuda')
model = torch.load(model_path)
testDataSet = SingleClassDataset(file_path="dataset/test_10")
testDataSet_loader = DataLoader(dataset=testDataSet, batch_size=1, shuffle=False)

f_txt = open('test.txt', 'a')
for i, image_low in enumerate(testDataSet_loader, 0):
    image_low = image_low[0].type(Tensor)
    image_low = Variable(image_low)
    r_h, g_h, b_h = model(image_low)
    #print(torch.sum(r_h))
    f_txt.write(str(r_h))
    f_txt.write('\n')
    f_txt.write(str(g_h))
    f_txt.write('\n')
    f_txt.write(str(b_h))
    f_txt.write('\n')

f_txt.close()

print('Finished')
