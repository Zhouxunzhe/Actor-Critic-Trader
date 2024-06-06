# Actor-Critic-Trader

This project uses Actor-Critic Deep Reinforcement Learning algorithms including A2C (Advantage Actor Critic), DDPG (Deep Deterministic Policy Gradient), and PPO (Proximal Policy Optimization) for portfolio management.

## Quickstart

1. Clone repo

   ```
   git clone https://github.com/Zhouxunzhe/Actor-Critic-Trader.git
   ```

2. Prepare conda env (assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed)

   Alternatively, you can skip this step and directly install on your Python env

   ```
   # We require cuda-11.3 python<=3.9
   conda create -n actrader python=3.9
   conda activate actrader
   ```

3. pip install requirements

   ```
   cd Actor-Critic-Trader
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. For cuda user (cuda 11.3)

   ```
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
5. Train the models

   ```
   python main.py
   ```

## Datasets

[Dow Jones Industrial Average (DJIA) 1990 - 2021](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average)

## **Result**

**A2C (Actor-Critic)** reaches the most profits at the 13 round.

| index               | stat     |
| ------------------- | -------- |
| Annual return       | 0.501005 |
| Cumulative returns  | 1.167534 |
| Annual volatility   | 0.313186 |
| Sharpe ratio        | 1.453883 |
| Calmar ratio        | 1.646764 |
| Stability           | 0.903189 |
| Max drawdown        | -0.30424 |
| Omega ratio         | 1.358105 |
| Sortino ratio       | 2.265669 |
| Skew                | 0.211108 |
| Kurtosis            | 13.59174 |
| Tail ratio          | 1.217694 |
| Daily value at risk | -0.03765 |

<img src="./assets/a2c_result.png" style="zoom:80%;" />

**DDPG (Deep Deterministic Policy Gradient)** reaches the most profits at the 14 round.

| index               | stat     |
| ------------------- | -------- |
| Annual return       | 0.591703 |
| Cumulative returns  | 1.423814 |
| Annual volatility   | 0.37695  |
| Sharpe ratio        | 1.422367 |
| Calmar ratio        | 1.88296  |
| Stability           | 0.864256 |
| Max drawdown        | -0.31424 |
| Omega ratio         | 1.299809 |
| Sortino ratio       | 2.132271 |
| Skew                | -0.01881 |
| Kurtosis            | 5.445235 |
| Tail ratio          | 1.014544 |
| Daily value at risk | -0.04536 |

<img src="./assets/ddpg_result.png" style="zoom:80%;" />

**PPO (Proximal Policy Optimization)** reaches the most profits at the 33 round.

| index               | stat     |
| ------------------- | -------- |
| Annual return       | 0.363893 |
| Cumulative returns  | 0.806028 |
| Annual volatility   | 0.274843 |
| Sharpe ratio        | 1.267362 |
| Calmar ratio        | 1.195302 |
| Stability           | 0.90252  |
| Max drawdown        | -0.30444 |
| Omega ratio         | 1.30618  |
| Sortino ratio       | 1.833882 |
| Skew                | -0.16044 |
| Kurtosis            | 12.17401 |
| Tail ratio          | 0.988949 |
| Daily value at risk | -0.03324 |

<img src="./assets/ppo_result.png" style="zoom:80%;" />

## Documentation

Browse the [project_report.pdf](./assets/project_report.pdf)

## License

Actor-Critic-Trader is MIT licensed. See the [LICENSE file](./LICENSE) for details.