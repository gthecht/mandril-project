from test import test

#%% Test Maml 10000
print(">> Maml:")
config_path = "maml-banditk5n10/config_maml_10000.json"
policy_path = "maml-banditk5n10/policy_maml_10000.th"
output_folder = "maml-banditk5n10/results_maml_10000.npz"
seed = 1
meta_batch_size = 20
num_batches = 10
num_workers = 8
device = "cpu"
alg = "maml"
test(
    config_path,
    output_folder,
    policy_path,
    seed=seed,
    num_workers=num_workers,
    num_batches=num_batches,
    meta_batch_size=meta_batch_size,
    device=device,
    alg=alg
)

#%% Test Mandril 10000
print(">> Mandril:")
config_path = "maml-banditk5n10/config_mandril_10000.json"
policy_path = "maml-banditk5n10/policy_mandril_10000.th"
output_folder = "maml-banditk5n10/results_mandril_10000.npz"
seed = 1
meta_batch_size = 20
num_batches = 10
num_workers = 8
device = "cpu"
alg = "mandril"
test(
    config_path,
    output_folder,
    policy_path,
    seed=seed,
    num_workers=num_workers,
    num_batches=num_batches,
    meta_batch_size=meta_batch_size,
    device=device,
    alg=alg
)
