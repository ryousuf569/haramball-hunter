import argparse

import torch

from config import EnvConfig, PPOConfig
from ppo import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PPOConfig.total_timesteps)
    p.add_argument("--out", default="checkpoint.pt")
    args = p.parse_args()

    actor, critic = train(PPOConfig(total_timesteps=args.steps), EnvConfig())
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, args.out)


if __name__ == "__main__":
    main()
