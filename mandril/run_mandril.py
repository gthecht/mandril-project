from train import train
import json

def run_mandril(output_folder, expert_type, expert_args):
    config = "configs/maml/bandit/bandit-k5-n10.yaml"
    seed = 1
    num_workers = 8
    device = "cpu"
    alg = "mandril"
    train(
        config,
        output_folder,
        seed,
        num_workers,
        device,
        alg,
        expert_type,
        expert_args,
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mandril')

    parser.add_argument('--output-folder', type=str,
        help='name of the output folder')

    parser.add_argument('--expert-type', type=str,
        help='type of the expert')

    parser.add_argument('--expert-args', type=str,
        help='type of the expert')

    args = parser.parse_args()

    output_folder = args.output_folder
    expert_type = args.expert_type
    expert_args = json.loads(args.expert_args)
    print(output_folder)
    print(expert_type)
    print(expert_args)

    run_mandril(output_folder, expert_type, expert_args)