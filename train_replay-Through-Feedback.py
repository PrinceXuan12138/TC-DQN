import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
#Replay-Through-Feedback（RTF）是一种基于生成重放的连续学习方法，由van de Ven等人于2022年提出1。RTF的基本思想是在训练新任务时，使用一个带有反馈连接的生成器来重放旧任务的数据，从而减少灾难性遗忘。

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

# 定义一个简单的变分自编码器
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = x.view(-1, 99) # 将图片展平为一维向量
        h = self.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 定义一个函数，用于计算变分自编码器的损失函数
def vae_loss(i,recon_x, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')
    # # recon_x=torch.sigmoid(recon_x)
    # # x=torch.sigmoid(x)
    # # print(recon_x.shape)
    # # print(x.shape)
    # try:
    #     recon_loss = bce(recon_x, x)
    # except Exception:
    #     print(i)
    #     print(recon_x.shape)
    #     print(x.shape)
    recon_loss = bce(recon_x, x.view(-1, 99))
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld_loss

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
    learning_rate = 0.00001
    rtf_lambda = 0.1 # RTF的正则系数
    replay_samples = 128 # 每个批次重放的样本数

    # 创建一个网络实例，一个变分自编码器实例和一个优化器
    net = Net(99, 100, 5)
    vae = VAE(99, 100, 20)
    optimizer = optim.SGD(list(net.parameters()) + list(vae.parameters()), lr=learning_rate)

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
    df.to_csv("classification_report_RTF_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (RTF method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_RTF_{}.png'.format(type_value))
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
            print(outputs.shape)
            print(labels.shape)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            if i==3:
                print('')
            # 生成重放数据
            replay_inputs = torch.randn(replay_samples, 20).to('cpu') # 从标准正态分布中采样隐变量
            replay_outputs = vae.decode(replay_inputs) # 通过变分自编码器生成重放数据
            replay_outputs=replay_outputs.unsqueeze(dim=1)
            replay_labels = net(replay_outputs) # 通过网络预测重放数据的标签
            replay_labels = torch.argmax(replay_labels, dim=1) # 取最大概率的类别作为重放数据的标签
            # 计算重放损失
            recon, mu, logvar = vae(replay_outputs) # 通过变分自编码器重建重放数据
            recon_loss = vae_loss(i,recon, replay_outputs, mu, logvar) # 计算重建损失
            pred_loss = nn.CrossEntropyLoss()(net(replay_outputs), replay_labels) # 计算预测损失
            replay_loss = recon_loss + rtf_lambda * pred_loss # 计算总的重放损失
            # 在损失函数中加上重放损失
            loss += replay_loss
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
    df.to_csv("classification_report_RTF_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (RTF method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_RTF_{}.png'.format(type_value))
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
    df.to_csv("classification_report_RTF_{}.csv".format(type_value), index=True)
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(testlabel, predict_label)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    plt.figure(figsize=(16, 16))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Traffic Classification Confusion Matrix (RTF method)")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.xticks(rotation=30)
    plt.yticks(rotation=45)
    plt.savefig('Confusion_RTF_{}.png'.format(type_value))
    acc=accuracy_score(testlabel,predict_label)
    print('Accuracy_score is {}'.format(acc))


