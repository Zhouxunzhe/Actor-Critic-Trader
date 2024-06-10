import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers, num_heads):
        super(ActorCritic, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout = 0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.permute(1, 0, 2)

        encoder_output = self.transformer_encoder(embedded_x)
        encoder_output = encoder_output.permute(1, 0, 2)

        encoder_output = encoder_output[:, -1, :]  # 取最后一个时间步的输出
        action_probs, state_value = self.actor(encoder_output), self.critic(encoder_output)
        return action_probs, state_value


# 自定义数据集
class StockDataset(Dataset):
    def __init__(self, csv_file, seq_length=10, end_date='2020-09-27'):
        self.data = pd.read_csv(csv_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        # 筛选训练数据
        self.data = self.data[self.data['Date'] <= pd.to_datetime(end_date)]

        self.data.set_index('Date', inplace=True)
        self.features = self.data[['Open', 'High', 'Low', 'Close']].values[:-1]
        self.targets = self.data['Close'].values[1:]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx:idx + self.seq_length - 1], dtype=torch.float32),
                torch.tensor(self.targets[idx + self.seq_length - 2], dtype=torch.float32))


# 定义奖励函数
def calculate_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_a2c(model, dataloader, optimizer, num_epochs, model_path, loss_file_path, gamma=0.99):
    model.train()
    criterion = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载先前的loss记录
    if os.path.exists(loss_file_path):
        loss_df = pd.read_csv(loss_file_path)
        start_epoch = len(loss_df)
    else:
        loss_df = pd.DataFrame(columns=['epoch', 'loss'])
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        state_memory = []
        action_memory = []
        reward_memory = []
        next_state_memory = []
        epoch_loss = 0

        for states, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{start_epoch + num_epochs}'):
            states, targets = states.to(device), targets.to(device)
            action_probs, state_values = model(states)

            actions = torch.multinomial(action_probs, 1).squeeze().tolist()
            action_memory.extend(actions)

            # reward = targets - states[:, -1, 3]  # 使用价格变化作为奖励
            # 使用价格变化作为奖励
            reward = torch.zeros_like(targets)
            for i, action in enumerate(actions):
                if action == 0:  # Buy
                    reward[i] = targets[i] - states[:, -1, 3][i]
                elif action == 1:  # Sell
                    reward[i] = states[:, -1, 3][i] - targets[i]
                elif action == 2:  # Hold
                    reward[i] = torch.tensor(0.0)
            reward_memory.extend(reward.tolist())

            next_states = states[:, 1:, :]  # 移动一个时间步
            next_state_values = model(next_states)[1]
            next_state_memory.extend(next_state_values.squeeze().tolist())

            state_memory.extend(state_values.squeeze().tolist())

            # 计算损失并更新模型
            if len(reward_memory) >= seq_length:
                rewards = calculate_returns(reward_memory, gamma)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                state_values = torch.tensor(state_memory, dtype=torch.float32).to(device)
                next_state_values = torch.tensor(next_state_memory, dtype=torch.float32).to(device)

                advantage = rewards - state_values + gamma * next_state_values

                critic_loss = criterion(state_values, rewards)
                action_probs_selected = action_probs.gather(1, torch.tensor(action_memory).unsqueeze(1).to(device))
                actor_loss = -torch.mean(torch.log(action_probs_selected) * advantage)

                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state_memory = []
                action_memory = []
                reward_memory = []
                next_state_memory = []

                epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')
        # print(f'Epoch {epoch + 1}, Rewards: {sum(total_rewards).item()}')

        # 保存模型
        if epoch % 10 == 0 or epoch == start_epoch + num_epochs - 1:
            torch.save(model.state_dict(), model_path)
            print(f"模型存储到 {model_path}")

        # 保存loss记录
        new_loss_record = pd.DataFrame({'epoch': [epoch + 1], 'loss': [avg_epoch_loss]})
        loss_df = pd.concat([loss_df, new_loss_record], ignore_index=True)
        loss_df.to_csv(loss_file_path, index=False)
        print(f"Loss record saved to {loss_file_path}")


if __name__ == "__main__":
    stock_name = "AAPL"
    csv_file = f'../data/{stock_name}.csv'
    model_path = f'../checkpoints/a2c_a2c_transformer/a2c_transformer_model_{stock_name}.pth'
    loss_file_path = f'../checkpoints/a2c_transformer/a2c_transformer_reward_{stock_name}.csv'
    learning_rate = 1e-4
    num_epochs = 150
    seq_length = 30
    input_dim = 4
    hidden_dim = 128
    action_dim = 3
    num_layers = 6  # Number of transformer layers
    num_heads = 8  # Number of attention heads

    dataset = StockDataset(csv_file, seq_length=seq_length, end_date='2021-06-06')
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    model = ActorCritic(input_dim, hidden_dim, action_dim, num_layers, num_heads)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 加载先前训练好的模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    train_a2c(model, dataloader, optimizer, num_epochs, model_path, loss_file_path)