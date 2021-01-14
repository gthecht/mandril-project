from train import train


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



#%% Train Perfect
print(">> Mandril with perfect expert:")
output_folder = "banditk5n10/perfect"
expert_type = "perfect"
expert_args = {}
run_mandril(output_folder, expert_type, expert_args)

#%% Train random from best 2
print(">> Mandril with random from best 2:")
output_folder = "banditk5n10/rand_from_2_best"
expert_type = "rand_from_k_best"
expert_args = {"k": 2}
run_mandril(output_folder, expert_type, expert_args)

#%% Train random from best 3
print(">> Mandril with random from best 3:")
output_folder = "banditk5n10/rand_from_3_best"
expert_type = "rand_from_k_best"
expert_args = {"k": 3}
run_mandril(output_folder, expert_type, expert_args)
