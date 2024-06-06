import warnings
from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp
import os
import torch
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device: %s' % device)


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for i in range(51):
        print(f"---------- round {i} ----------")

        if not os.path.isfile(f'plots/ddpg/{i}2_testing.png'):
            ddpg = DDPG(state_type='indicators', djia_year=2019, repeat=i)
            # ddpg.train()
            ddpg.test()
            del ddpg
            torch.cuda.empty_cache()

        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            # ppo.train()
            ppo.test()
            del ppo
            torch.cuda.empty_cache()

        if not os.path.isfile(f'plots/a2c/{i}2_testing.png'):
            a2c = A2C(n_agents=4, state_type='indicators', djia_year=2019, repeat=i)
            # a2c.train()
            a2c.test()
            del a2c
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
