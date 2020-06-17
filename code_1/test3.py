# import os
# import numpy
# path = r'D:\python\pycharm\1\histogram\train_r.txt'
# # his_path=os.listdir(path)
# # # high_his = os.path.join(his_path, his_path[index])
# f = open(path)
# lines = f.readlines()
# lines = numpy.array(lines)
# print(lines)

# from torch.utils.data import Dataset
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

from simplenet import SqueezeNet


# from net import SqueezeNet
class SingleClassDataset(Dataset):

    def __init__(self, file_path):
        # 保证输入的是正确的路径
        if not os.path.isdir(file_path):
            raise ValueError("input file_path is not a dir")
        self.file_path = file_path
        self.file_path_high = os.path.join(self.file_path, 'high/')
        self.file_path_low = os.path.join(self.file_path, 'low/')
        self.file_path_his = os.path.join(self.file_path,'histogram/')
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
        file_path_his_r = os.path.join(self.file_path_his, 'r')
        file_path_his_r = file_path_his_r+ str(index)+'.txt'
        file_path_his_g = os.path.join(self.file_path_his, 'g')
        file_path_his_g = file_path_his_g + str(index) + '.txt'
        file_path_his_b = os.path.join(self.file_path_his, 'b')
        file_path_his_b = file_path_his_b + str(index) + '.txt'
        f_r = np.loadtxt(file_path_his_r)
        his_r = torch.from_numpy(f_r)
        f_g = np.loadtxt(file_path_his_g)
        his_g = torch.from_numpy(f_g)
        f_b = np.loadtxt(file_path_his_b)
        his_b = torch.from_numpy(f_b)
        return (image_high, image_low,his_r,his_g,his_b)

    def _read_convert_image(self, image_name):
        image = Image.open(image_name)
        # image = self.transforms(image).float()

        tf_resize = transforms.Compose([transforms.Resize((224, 224))])
        image = tf_resize(image)
        image = T.ToTensor()(image)
        # image = torch.reshape(image,(-1,3,224,224))
        return image

    def __len__(self):
        return len(self.image_list_high)


#cuda = True if torch.cuda.is_available() else False
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#print(torch.version.cuda)
#print(torch.cuda.is_available())
Tensor = torch.cuda.FloatTensor
print(Tensor)
device1 = torch.device('cuda')
model = SqueezeNet().to(device1)
trainDataSet = SingleClassDataset(file_path="dataset/train")
trainDataSet_loader = DataLoader(dataset=trainDataSet, batch_size=1, shuffle=True)
testDataSet = SingleClassDataset(file_path="dataset/test")
testDataSet_loader = DataLoader(dataset=testDataSet, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-9, weight_decay=0.1)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

train_loss = []
test_loss = []

# model_path = "log/model.pkl"
# torch.save(model, model_path)

for epoch in range(200):
   # f = open('trainloss.txt', 'a')
    running_loss = 0.0
    for i, (image_high, image_low,his_r,his_g,his_b) in enumerate(trainDataSet_loader, 0):

        image_high = image_high.type(Tensor)
        image_low = image_low.type(Tensor)
        his_r = his_r.type(Tensor)
        his_g = his_g.type(Tensor)
        his_b = his_b.type(Tensor)
        image_high = Variable(image_high,requires_grad=True)
        image_low = Variable(image_low,requires_grad=True)
        optimizer.zero_grad()
        r_h, g_h, b_h = model(image_low)
        image_high = torch.squeeze(image_high)
        high_r = his_r / 50176
        high_g = his_g / 50176
        high_b = his_b / 50176
        criterion = torch.cosine_similarity
        loss_r = 1-criterion(r_h, high_r)
        loss_g = 1-criterion(g_h, high_g)
        loss_b = 1-criterion(b_h, high_b)
        loss = (loss_r + loss_g + loss_b )/3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i == 379:
            print('[epoch:%d] train loss: %.11f' %
                  (epoch + 1, running_loss / 380))
            train_loss.append(running_loss / 380)
            running_loss = 0.0
    #f.write('\n'+list(train_loss))
    test_running_loss = 0.0
    for i, (image_high, image_low,his_r,his_g,his_b) in enumerate(testDataSet_loader, 0):
        image_high = image_high.type(Tensor)
        image_low = image_low.type(Tensor)
        his_r = his_r.type(Tensor)
        his_g = his_g.type(Tensor)
        his_b = his_b.type(Tensor)
        image_high = Variable(image_high,requires_grad=True)
        image_low = Variable(image_low,requires_grad=True)
        r_h, g_h, b_h = model(image_low)

        image_high = torch.squeeze(image_high)
        high_r = his_r / 50176
        # print(high_r)
        # high_r = torch.unsqueeze(high_r, 0)
        high_g = his_g / 50176
        # high_g = torch.unsqueeze(high_g, 0)
        high_b = his_b / 50176
        # print(high_r.size(),high_g.size(),high_b.size())
        # print(high_r)
        criterion = torch.cosine_similarity
        loss_r = 1-criterion(r_h, high_r)
        loss_g = 1-criterion(g_h, high_g)
        loss_b = 1-criterion(b_h, high_b)
        #loss_hist = 3 - (criterion(r_h, g_h) + criterion(r_h, b_h) + criterion(g_h, b_h))
        # loss_r_res = criterion(r_h,r_res)
        # loss_g_res = criterion(g_h,g_res)
        # loss_b_res = criterion(b_h,b_res)
        loss = (loss_r + loss_g + loss_b ) / 3
        # loss.backward()
        # optimizer.step()
        test_running_loss += loss.item()
        if i == 93:
            print('[epoch:%d] test loss: %.11f' %
                  (epoch + 1, test_running_loss / 94))
            test_loss.append(test_running_loss / 94)
            test_running_loss = 0.0
    if epoch == 9 or epoch == 19 or epoch == 49 or epoch == 99:
        model_path = "log/model_" + str(epoch + 1) + ".pkl"
        torch.save(model, model_path)
        f_loss = open('loss.txt', 'a')
        for i in range(len(train_loss)):
            f_loss.write(str(train_loss[i]) + ' ')
        f_loss.write('\n')
        for i in range(len(test_loss)):
            f_loss.write(str(test_loss[i]) + ' ')
        f_loss.close()

model_path = "log/model_" + ".pkl"
torch.save(model, model_path)

print('Finished')

