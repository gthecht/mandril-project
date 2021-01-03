import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

from mazelab import MazeEnv

# Register env:
gym.envs.register(
    id="RandomShapeMaze-v0",
    entry_point=MazeEnv,
    max_episode_steps=200,
    kwargs={"width": 50, "height": 50, "max_shapes": 50,
            "max_size": 8, "allow_overlap": False, "shape": None}
)

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'env-kwargs' not in config.keys(): config['env-kwargs'] = {}

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': [], "train_episodes": [], "valid_episodes": []}
    train_returns, valid_returns = [], []
    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(
            tasks,
            num_steps=config['num-steps'],
            fast_lr=config['fast-lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae-lambda'],
            device=args.device
        )

        logs['tasks'].extend(tasks)
        logs['train_episodes'].extend(train_episodes)
        logs['valid_episodes'].extend(valid_episodes)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        logs['train_returns'] = np.concatenate(train_returns, axis=0)
        logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

        with open(args.output, 'wb') as f:
            np.savez(f, **logs)


if __name__ == '__main__':
    torch.set_num_threads(1)

    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    args = parser.parse_args()
    args.config = "maml-randomShapeMaze/config.json"
    args.policy = "maml-randomShapeMaze/policy.th"

    # Evaluation
    args.num_batches = 10
    args.meta_batch_size = 40

    # Miscellaneous
    args.output = "maml-randomShapeMaze/results.npz"
    args.seed = None
    args.num_workers = 8
    args.device = ('cpu')

    main(args)
