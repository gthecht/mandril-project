from test import test
import json

def test_mandril(model_folder, expert_type, expert_args):
    seed = 1
    num_workers = 8
    meta_batch_size = 20
    num_batches = 100
    num_workers = 8
    device = "cpu"
    alg = "mandril"

    config_path = model_folder + "/config.json"
    policy_path = model_folder + "/policy.th"
    output_path = model_folder + "/results.npz"

    test(
        config_path,
        output_path,
        policy_path,
        seed=seed,
        num_workers=num_workers,
        num_batches=num_batches,
        meta_batch_size=meta_batch_size,
        device=device,
        alg=alg,
        expert_type=expert_type,
        expert_args=expert_args,
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

    test_mandril(output_folder, expert_type, expert_args)