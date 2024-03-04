import os
import pandas as pd
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model, Sequential,load_model
import keras
import keras.backend as K
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from collections import deque
import random,time

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# 1 是旧训练旧测
# 2 是新训练旧测
# 3 是新训练新测
type_value='1'

data = pd.read_csv('../csv_data/iscx_csv_deal_map.csv')
data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()

data_x = data.iloc[:, :-1]
data_y = data['appname']
# train & test
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.8,random_state=5)
x_train=x_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

num_actions = len(set(y_test))
input_size=data_x.shape[1]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)


dummy_actions = np.ones((1, num_actions))
y_train_onehot = keras.utils.to_categorical(y_train, num_actions)

y_test_onehot = keras.utils.to_categorical(y_test, num_actions)


class MnEnviroment(object):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self._sample_index()
        self.action_space = len(set(y)) - 1

    def reset(self):
        obs, _ = self.step(-1)
        return obs

    '''
    action: 0-9 categori, -1 : start and no reward
    return: next_state(image), reward
    '''

    def step(self, action):
        if action == -1:
            _c_index = self.current_index
            self.current_index = self._sample_index()
            return (self.train_X[_c_index], 0)
        r = self.reward(action)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index], r

    def reward(self, action):
        c = self.train_Y[self.current_index]
        # print(c)
        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space)

    def _sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)




env = MnEnviroment(x_train, y_train)
memory = deque(maxlen=512)
replay_size = 64
epoches = 20
pre_train_num = 256
gamma = 0.    #every state is i.i.d
alpha = 0.5
forward = 512
epislon_total = 2018



def createDQN(input_size, actions_num):
    img_input = Input(shape=(input_size, 1), dtype='float32', name='image_inputs')
    # conv1
    conv1 = Conv1D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(img_input)
    conv2 = Conv1D(64, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling1D(pool_size=(2))(conv2)

    conv3 = Conv1D(128, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv4 = Conv1D(128, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    pool2 = MaxPooling1D(pool_size=(2))(conv4)

    x = Flatten()(pool2)
    x = Dense(128, activation='relu')(x)
    outputs_q = Dense(actions_num, name='q_outputs')(x)
    # one hot input
    actions_input = Input((actions_num,), name='actions_input')
    q_value = multiply([actions_input, outputs_q])
    q_value = Lambda(lambda l: K.sum(l, axis=1, keepdims=True), name='q_value')(q_value)

    model = Model(inputs=[img_input, actions_input], outputs=q_value)
    model.compile(loss='mse', optimizer='adam')
    return model


actor_model = createDQN(input_size,num_actions) #用于决策
critic_model = createDQN(input_size,num_actions) #用于训练

actor_q_model = Model(inputs=actor_model.input, outputs=actor_model.get_layer('q_outputs').output)





def copy_critic_to_actor():
    critic_weights = critic_model.get_weights()
    actor_wegiths  = actor_model.get_weights()
    for i in range(len(critic_weights)):
        actor_wegiths[i] = critic_weights[i]
    actor_model.set_weights(actor_wegiths)

def get_q_values(model_,state):
    inputs_ = [state.reshape(1,*state.shape),dummy_actions]
    qvalues = model_.predict(inputs_)
    return qvalues[0]

def predict(model,states):
    inputs_ = [states, np.ones(shape=(len(states),num_actions))]
    qvalues = model.predict(inputs_)
    return np.argmax(qvalues,axis=1)

def epsilon_calc(step, ep_min=0.01,ep_max=1,ep_decay=0.0001,esp_total = 1000):
    return max(ep_min, ep_max -(ep_max - ep_min)*step/esp_total )

def epsilon_greedy(env, state, step, ep_min=0.01, ep_decay=0.0001,ep_total=1000):
    epsilon = epsilon_calc(step, ep_min, 1, ep_decay,ep_total)
    if np.random.rand()<epsilon:
        return env.sample_actions(),0
    qvalues = get_q_values(actor_q_model, state)
    return np.argmax(qvalues), np.max(qvalues)

def pre_remember(pre_go = 30):
    state = env.reset()
    for i in range(pre_go):
        rd_action = env.sample_actions()
        next_state, reward = env.step(rd_action)
        remember(state,rd_action,0,reward,next_state)
        state = next_state


def remember(state, action, action_q, reward, next_state):
    memory.append([state, action, action_q, reward, next_state])


def sample_ram(sample_num):
    return np.array(random.sample(memory, sample_num))


def replay():
    if len(memory) < replay_size:
        return
        # 从记忆中i.i.d采样
    samples = sample_ram(replay_size)
    # 展开所有样本的相关数据
    # 这里next_states没用 因为和上一个state无关。
    states, actions, old_q, rewards, next_states = zip(*samples)
    states, actions, old_q, rewards = np.array(states), np.array(actions).reshape(-1, 1), \
                                      np.array(old_q).reshape(-1, 1), np.array(rewards).reshape(-1, 1)

    actions_one_hot = keras.utils.to_categorical(actions, num_actions)
    # print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
    # 从actor获取下一个状态的q估计值 这里也没用 因为gamma=0 也就是不对bellman方程展开
    # inputs_ = [next_states,np.ones((replay_size,num_actions))]
    # qvalues = actor_q_model.predict(inputs_)

    # q = np.max(qvalues,axis=1,keepdims=True)
    q = 0
    q_estimate = (1 - alpha) * old_q + alpha * (rewards.reshape(-1, 1) + gamma * q)
    history = critic_model.fit([states, actions_one_hot], q_estimate, epochs=1, verbose=0)
    return np.mean(history.history['loss'])



if type_value!='1':
    if not os.path.exists("../log/DQN_model_3.h5"):
        old_model=load_model("../log/DQN_model_1.h5")
        critic_model.set_weights(old_model.get_weights())
        copy_critic_to_actor()
    else:
        old_model=load_model("../log/DQN_model_3.h5")
        critic_model.set_weights(old_model.get_weights())
        copy_critic_to_actor()

if type_value=='1' or not os.path.exists("../log/DQN_model_3.h5"):
    memory.clear()
    total_rewards = 0
    reward_rec = []
    pre_remember(pre_train_num)
    every_copy_step = 128

    pbar = tqdm(range(1,epoches+1))
    state = env.reset()
    for epoch in pbar:
        total_rewards = 0
        epo_start = time.time()
        for step in range(forward):
            #对每个状态使用epsilon_greedy选择
            action, q = epsilon_greedy(env, state, epoch, ep_min=0.01, ep_total=epislon_total)
            eps = epsilon_calc(epoch,esp_total=epislon_total)
            #play
            next_state,reward = env.step(action)
            #加入到经验记忆中
            remember(state, action, q, reward, next_state)
            #从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
            loss = replay()
            total_rewards += reward
            state = next_state
            if step % every_copy_step==0:
                copy_critic_to_actor()
        reward_rec.append(total_rewards)
        pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards,loss,int(time.time()-epo_start),eps))
    critic_model.save('../log/DQN_model_{}.h5'.format(type_value))
# r5 = np.mean([reward_rec[i:i+10] for i in range(0,len(reward_rec),10)],axis=1)
# plt.plot(range(len(r5)),r5,c='b')
# plt.xlabel('iters')
# plt.ylabel('mean score')
# plt.show()

# 评估模型
print('开始模型评估-------------------------------------------------')
# scores = actor_q_model.evaluate(x_test, y_test, verbose=1)

y_pred = predict(actor_q_model, x_test)

# 输出report
# labels = ['Short video','Long video','Moba game','Shooting game','Music','Social media','Live stream']
labels = ['chat', 'file', 'email', 'streaming', 'voip']
y_test=y_test.values
report = classification_report(y_test, y_pred, target_names=labels, digits=4,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_DQN_{}.csv".format(type_value), index=True)

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
plt.title("Traffic Classification Confusion Matrix (DQN method)")
plt.ylabel('Application traffic samples')
plt.xlabel('Application traffic samples')
plt.xticks(rotation=30,fontsize=25)
plt.yticks(rotation=45,fontsize=25)
plt.savefig('Confusion_DQN_{}.png'.format(type_value))
acc=accuracy_score(y_test,y_pred)
print('Accuracy_score is {}'.format(acc))