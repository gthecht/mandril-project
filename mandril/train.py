import os, sys
parentdir = os.path.dirname(os.path.realpath("./mandril"))
sys.path.append(parentdir)

import gym
import torch
import json
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def train(config_path, output_folder, seed=None, num_workers=1, device="cpu", alg="maml"):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        policy_filename = os.path.join(output_folder, 'policy.th')
        config_filename = os.path.join(output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config["config"] = config_path
            config["output_folder"] = output_folder
            config["seed"] = seed
            config["num_workers"] = num_workers
            config["use_cuda"] = device == "cuda"
            config["device"] = device
            json.dump(config, f, indent=2)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=seed,
                               alg=alg,
                               num_workers=num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=device,
                           alg=alg)

    num_iterations = 0
    for batch in trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        # >> need to change the futures to use an expert's demos
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        # Save policy
        if output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    config = "configs/maml/bandit/bandit-k5-n10.yaml"
    output_folder = "maml-banditk5n10"
    seed = 1
    num_workers = 8
    device = "cpu"
    alg = "maml"
    train(config, output_folder, seed, num_workers, device, alg)
