from test import test

def test_mandril(model_folder, expert_type, expert_args):
    config_path = "configs/maml/bandit/bandit-k5-n10.yaml"
    seed = 1
    num_workers = 8
    meta_batch_size = 20
    num_batches = 100
    num_workers = 8
    device = "cpu"
    alg = "mandril"

    output_folder = model_folder + "config.json"
    policy_path = model_folder + "policy.json"

    test(
        config_path,
        output_folder,
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

#%% Test Perfect
print(">> Mandril with perfect expert:")
model_folder = "banditk5n10/perfect"
expert_type = "perfect"
expert_args = {}
test_mandril(model_folder, expert_type, expert_args)

#%% Test random from best 2
print(">> Mandril with random from best 2:")
model_folder = "banditk5n10/rand_from_2_best"
expert_type = "rand_from_k_best"
expert_args = {"k": 2}
test_mandril(model_folder, expert_type, expert_args)

#%% Test random from best 3
print(">> Mandril with random from best 3:")
model_folder = "banditk5n10/rand_from_3_best"
expert_type = "rand_from_k_best"
expert_args = {"k": 3}
test_mandril(model_folder, expert_type, expert_args)
