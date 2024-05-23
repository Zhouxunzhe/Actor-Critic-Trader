# Actor_Critic_Trader

This project uses Actor-Critic Deep Reinforcement Learning algorithms including A2C (Advantage Actor Critic), DDPG (Deep Deterministic Policy Gradient), and PPO (Proximal Policy Optimization) for portfolio management.

## Quickstart

1. Clone repo

   ```
   git clone https://github.com/Zhouxunzhe/Actor-Critic-Trader.git
   ```

2. Prepare conda env (assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed)

   Alternatively, you can skip this step and directly install on your Python env

   ```
   # We require python<=3.8
   conda create -n actrader python=3.8
   conda activate actrader
   ```

3. pip install requirements

   ```
   cd Actor-Critic-Trader
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. train the models

   ```
   
   ```

## Datasets

[Dow Jones Industrial Average (DJIA) 2012 - 2019](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average)

## **Result**

**A2C (Actor-Critic)** reaches profits _ at _ round, against _ of buy&hold.

![](./plots/a2c/12_testing.png)

**DDPG (Deep Deterministic Policy Gradient)** reaches profits _ at _ round, against _ of buy&hold.

![](./plots/ddpg/92_testing.png)

**PPO (Proximal Policy Optimization)** reaches profits _ at _ round, against _ of buy&hold.

![](./plots/ppo/42_testing.png)

## Conclusion

## Documentation

Browse the [project_report.pdf](./project_report.pdf)

## License

Actor-Critic-Trader is MIT licensed. See the [LICENSE file](./LICENSE) for details.