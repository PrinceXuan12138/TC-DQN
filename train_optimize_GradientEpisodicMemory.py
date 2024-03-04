import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Gradient Episodic Memory（GEM）是一种解决神经网络灾难性遗忘问题的方法，由Lopez-Paz和Ranzato于2017年提出1。GEM的基本思想是在训练新任务时，保证梯度的方向不会降低旧任务的性能
# 定义一个简单的全连接网络
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = x.view(-1, 99) # 将图片展平为一维向量
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
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

# # 定义一个自定义的数据集类
# class MyDataset(Dataset):
#     # 初始化，读取csv文件
#     def __init__(self, csv_file):
#         self.data = []
#         self.label = []
#         with open(csv_file, 'r') as f:
#             reader = csv.reader(f)
#             next(reader)
#             for row in reader:
#                 # 假设数据的最后一列是标签，其他列是特征
#                 self.data.append([float(x) for x in row[:-1]])
#                 self.label.append(int(row[-1]))
#
#         # 将列表转换为张量
#         self.data = torch.tensor(self.data)
#         self.label = torch.tensor(self.label)
#
#     # 返回数据集的大小
#     def __len__(self):
#         return len(self.data)
#
#     # 根据索引返回一条数据和标签
#     def __getitem__(self, index):
#         return self.data[index].view(1, 99), self.label[index]


# # 定义两个不同的MNIST数据集，第二个数据集将图片旋转90度
# transform1 = transforms.ToTensor()
# transform2 = transforms.Compose([transforms.RandomRotation((90, 90)), transforms.ToTensor()])
# dataset1 = datasets.MNIST(root='./data', train=True, download=True, transform=transform1)
# dataset2 = datasets.MNIST(root='./data', train=True, download=True, transform=transform2)
# loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
# loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

# 定义一个函数，用于更新记忆
def update_memory(loader, task_id):
    global gem_memory
    # 如果记忆列表的长度小于任务编号，说明是一个新任务，需要添加一个新的记忆
    if len(gem_memory) < task_id + 1:
        gem_memory.append({'inputs': [], 'labels': []})
    # 从数据加载器中随机采样记忆大小的数据
    inputs, labels = next(iter(loader))
    # 将数据转换为张量并存储到记忆中
    gem_memory[task_id]['inputs'] = inputs.to('cpu')
    gem_memory[task_id]['labels'] = labels.to('cpu')

# 定义一个函数，用于计算GEM的约束
def compute_gem_constraint():
    global gem_memory
    # 初始化一个梯度列表
    grad_list = []
    # 遍历每个任务的记忆
    for task_id, memory in enumerate(gem_memory):
        # 获取记忆中的输入和标签
        inputs = memory['inputs']
        labels = memory['labels']
        # 通过网络预测输出
        outputs = net(inputs)
        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # 对损失求梯度
        net.zero_grad()
        loss.backward()
        # 将网络参数的梯度存储到列表中
        grad_list.append([])
        for param in net.parameters():
            grad_list[task_id].append(param.grad.clone().detach())
    # 初始化一个约束列表
    constraint_list = []
    # 遍历网络参数
    for param_index, param in enumerate(net.parameters()):
        # 初始化一个约束张量
        constraint = torch.zeros_like(param.grad)
        # 遍历每个任务的梯度
        for task_id, grad in enumerate(grad_list):
            # 计算当前梯度和记忆梯度的内积
            dot_product = torch.sum(grad[param_index] * param.grad)
            # 如果内积小于边界系数，说明当前梯度会降低记忆任务的性能，需要进行约束
            if dot_product < gem_margin:
                # 计算记忆梯度的范数
                norm = torch.sum(grad[param_index] ** 2)
                # 如果范数不为零，将记忆梯度的方向加到约束张量上
                if norm > 0:
                    constraint += (gem_margin - dot_product) / norm * grad[param_index]
        # 将约束张量存储到列表中
        constraint_list.append(constraint)
    # 返回约束列表
    return constraint_list


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
    # 创建一个网络实例和一个优化器
    net = Net(99, 100, 5)
    # 定义一些超参数
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.0001
    gem_memory_size = 256 # GEM的记忆大小
    gem_margin = 0.5 # GEM的边界系数
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)


    # 定义一个列表，用于存储不同任务的记忆
    gem_memory = []

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
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("Finished training on task 1.")

    # 更新第一个任务的记忆
    update_memory(origin_train_loader, 0)

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
    df.to_csv("classification_report_GEM_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (GEM method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_GEM_{}.png'.format(type_value))
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
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # 获取GEM的约束
            constraint_list = compute_gem_constraint()
            # 将约束加到网络参数的梯度上
            for param_index, param in enumerate(net.parameters()):
                param.grad += constraint_list[param_index]
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
    df.to_csv("classification_report_GEM_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (GEM method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_GEM_{}.png'.format(type_value))
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
    df.to_csv("classification_report_GEM_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (GEM method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_GEM_{}.png'.format(type_value))
    acc=accuracy_score(testlabel,predict_label)
    print('Accuracy_score is {}'.format(acc))