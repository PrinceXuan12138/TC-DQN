import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
#Selfless Sequential Learning（SSL）是一种解决神经网络灾难性遗忘问题的方法，由Aljundi等人于2019年提出1。
# SSL的基本思想是在训练新任务时，保证网络的容量不会被当前任务占满，而是留出一部分空间给未来的任务2。为了实现SSL，
# 作者提出了一种新的正则化方法，叫做Selfless Lateral Neural Inhibition（SLNI），它通过抑制神经元的激活来增加网络的稀疏性2

# 定义一个简单的全连接网络
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = x.view(-1, 99) # 将图片展平为一维向量
#         x = self.relu(self.fc1(x))
#         x =self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        # self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        # self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        # # self.conv4 = nn.Conv1d(32, 32, 3, padding=1)
        # self.pool2 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

        self.fc3 = nn.Linear(64 * 49 , output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.pool1(x)
        # x = F.relu(self.conv3(x))
        # x = self.pool2(x)
        # x = F.relu(self.conv4(x))
        x = self.flatten(x)

        x = self.fc3(x)
        return x


# 定义一个函数，用于计算SLNI的正则项
def compute_slni_loss():
    loss = 0
    # 遍历网络的每一层
    for layer in net.children():
        # 如果是全连接层，获取其权重矩阵
        if isinstance(layer, nn.Linear):
            weight = layer.weight
            # 计算权重矩阵的每一行的范数
            norm = torch.norm(weight, dim=1, keepdim=True)
            # 对范数进行排序，获取排序后的索引
            _, indices = torch.sort(norm, dim=0, descending=True)
            # 对索引进行反转，使得范数最小的行在最前面
            indices = torch.flip(indices, dims=[0])
            # 根据索引重新排列权重矩阵
            weight = weight[indices]
            # 计算每一行与其邻域内的行的内积
            for i in range(weight.size(0)):
                for j in range(max(0, i - slni_radius), min(weight.size(0), i + slni_radius + 1)):
                    if i != j:
                        loss += torch.sum(weight[i] * weight[j])
    # 返回正则项
    return loss

# 定义一个自定义的数据集类
class MyDataset(Dataset):
    # 初始化，读取csv文件
    def __init__(self, Xdata,Ylabel):
        self.data = []
        self.label = []
        for row in Xdata:
            self.data.append([float(x) for x in row])
        for row in Ylabel:
            self.label.append(int(row))
        # 将列表转换为张量
        self.data = torch.tensor(self.data)
        self.label = torch.tensor(self.label)

    # 返回数据集的大小
    def __len__(self):
        return len(self.data)

    # 根据索引返回一条数据和标签
    def __getitem__(self, index):
        return self.data[index].view(1, 98), self.label[index]

if __name__ == "__main__":
    # 定义一些超参数
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.0001
    slni_lambda = 0.1 # SLNI的正则系数
    slni_radius = 5 # SLNI的邻域半径

    # 创建一个网络实例和一个优化器
    net = Net(99, 100, 5)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    #step1开始加载数据集
    origindata = pd.read_csv('./csv_data/old_deal_split_std.csv')
    origindata = origindata.replace([np.inf, -np.inf], np.nan).dropna().copy()
    data_x = origindata.iloc[:, :-1]
    data_y = origindata['appname']


    origin_x_train, origin_x_test, origin_y_train, origin_y_test = train_test_split(data_x, data_y, train_size=0.8,random_state=5)
    # scaler = StandardScaler()
    # origin_x_train = scaler.fit_transform(origin_x_train)
    # origin_x_test = scaler.fit_transform(origin_x_test)
    origin_x_train=origin_x_train.values
    origin_x_test=origin_x_test.values

    
    origin_train_dataset= MyDataset(origin_x_train,origin_y_train.values)
    origin_train_loader = DataLoader(origin_train_dataset, batch_size=batch_size, shuffle=True)
    origin_test_dataset= MyDataset(origin_x_test,origin_y_test.values)
    origin_test_lodaer = DataLoader(origin_test_dataset, batch_size=batch_size, shuffle=True)



    driftdata= pd.read_csv('./csv_data/new_deal_split_std.csv')
    data_x = driftdata.iloc[:, :-1]
    data_y = driftdata['appname']


    drift_x_train, drift_x_test, drift_y_train, drift_y_test = train_test_split(data_x, data_y, train_size=0.8,random_state=5)
    # scaler = StandardScaler()
    # drift_x_train = scaler.fit_transform(drift_x_train)
    # drift_x_test = scaler.fit_transform(drift_x_test)

    drift_x_train=drift_x_train.values
    drift_x_test=drift_x_test.values

    drift_train_dataset= MyDataset(drift_x_train,drift_y_train.values)
    drift_train_loader = DataLoader(drift_train_dataset, batch_size=batch_size, shuffle=True)

    drift_test_dataset= MyDataset(drift_x_test,drift_y_test.values)
    drift_test_lodaer = DataLoader(drift_test_dataset, batch_size=batch_size, shuffle=True)



# 训练第一个任务
    print("Training on task 1...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(origin_train_loader):
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels) + slni_lambda * compute_slni_loss() # 在损失函数中加上SLNI的正则项
            # loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("Finished training on task 1.")

    #获取测试结果
    print("Testing on task 1 using origindata ...")
    type_value='1'
    with torch.no_grad():
        test_pred = []
        test_true=[]
        for i, (inputs, labels) in enumerate(origin_test_lodaer):
            outputs = net(inputs)
            test_pred.append(outputs)
            test_true.append(labels)
        test_pred = torch.cat(test_pred)
        test_true = torch.cat(test_true)
        _, pre_label = torch.max(test_pred, dim=1)

        predict_label = pre_label.cpu().numpy()
        testlabel = test_true.cpu().numpy()

    acc = accuracy_score(testlabel, predict_label)
    print(acc)
    # labels = ['Short video','Long video','Moba game','Shooting game','Music','Social media','Live stream']
    labels = ['chat','file','email','streaming','voip']
    report=classification_report(testlabel, predict_label, target_names=labels,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("classification_report_SSL_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (SSL method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_SSL_{}.png'.format(type_value))
    acc=accuracy_score(testlabel,predict_label)
    print('Accuracy_score is {}'.format(acc))




    # 训练第二个任务
    print("Training on task 2...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(drift_train_loader):
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels) + slni_lambda * compute_slni_loss() # 在损失函数中加上SLNI的正则项
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("Finished training on task 2.")

    print("Testing on task 2 using origindata ...")
    type_value='2'
    with torch.no_grad():
        test_pred = []
        test_true=[]
        for i, (inputs, labels) in enumerate(origin_test_lodaer):
            outputs = net(inputs)
            test_pred.append(outputs)
            test_true.append(labels)
        test_pred = torch.cat(test_pred)
        test_true = torch.cat(test_true)
        _, pre_label = torch.max(test_pred, dim=1)

        predict_label = pre_label.cpu().numpy()
        testlabel = test_true.cpu().numpy()

    acc = accuracy_score(testlabel, predict_label)
    print(acc)
    labels = ['Short video','Long video','Moba game','Shooting game','Music','Social media','Live stream']
    report=classification_report(testlabel, predict_label, target_names=labels,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("classification_report_SSL_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (SSL method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_SSL_{}.png'.format(type_value))
    acc=accuracy_score(testlabel,predict_label)
    print('Accuracy_score is {}'.format(acc))

    print("Testing on task 2 using diftdata ...")
    type_value='3'
    with torch.no_grad():
        test_pred = []
        test_true=[]
        for i, (inputs, labels) in enumerate(drift_test_lodaer):
            outputs = net(inputs)
            test_pred.append(outputs)
            test_true.append(labels)
        test_pred = torch.cat(test_pred)
        test_true = torch.cat(test_true)
        _, pre_label = torch.max(test_pred, dim=1)

        predict_label = pre_label.cpu().numpy()
        testlabel = test_true.cpu().numpy()
    acc = accuracy_score(testlabel, predict_label)
    print(acc)
    labels = ['Short video','Long video','Moba game','Shooting game','Music','Social media','Live stream']
    report=classification_report(testlabel, predict_label, target_names=labels,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("classification_report_SSL_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (SSL method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_SSL_{}.png'.format(type_value))
    acc=accuracy_score(testlabel,predict_label)
    print('Accuracy_score is {}'.format(acc))