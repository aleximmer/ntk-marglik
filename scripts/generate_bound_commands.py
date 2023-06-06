# Illustration of bounds
base_cmds = ['classification_image.py --config configs/bound_illustration.yaml']
seeds = [1, 2, 3]
for base_cmd in base_cmds:
    for seed in seeds:
        cmd = base_cmd + f' --seed {seed}'
        # baseline MAP
        print(cmd, '--n_epochs_burnin 1000 --approx kron --method baseline')
        for batch_size in [10, 20, 50, 100, 250, 500, 1000]:
            cmd = base_cmd + f' --seed {seed} --marglik_batch_size {batch_size} --bound'
            for approx in ['full', 'blockdiag', 'kron', 'diag', 'kernel']:
                print(cmd, '--approx', approx)
                print(cmd, '--approx', approx, '--grouped_loader')
                print(cmd, '--approx', approx, '--independent_outputs', '--single_output')
                print(cmd, '--approx', approx, '--independent_outputs', '--single_output', '--grouped_loader')

# Illustration of bounds with lila
base_cmds = ['classification_image.py --config configs/bound_illustration_lila.yaml']
seeds = [1, 2, 3]
for base_cmd in base_cmds:
    for seed in seeds:
        cmd = base_cmd + f' --seed {seed}'
        # baseline MAP
        print(cmd, '--n_epochs_burnin 1000 --approx kron --method baseline')
        for batch_size in [10, 20, 50, 100, 250, 500, 1000]:
            cmd = base_cmd + f' --seed {seed} --marglik_batch_size {batch_size} --bound'
            for approx in ['full', 'blockdiag', 'kron', 'diag', 'kernel']:
                print(cmd, '--approx', approx)
                print(cmd, '--approx', approx, '--grouped_loader')
                print(cmd, '--approx', approx, '--independent_outputs', '--single_output')
                print(cmd, '--approx', approx, '--independent_outputs', '--single_output', '--grouped_loader')

# TIMING commands
base_cmd = 'classification_image.py --config configs/bound_illustration_timing.yaml'
for batch_size in [10, 20, 50, 100, 250, 500, 1000]:
    cmd = base_cmd + f' --marglik_batch_size {batch_size} --bound'
    for approx in ['full', 'kron', 'diag', 'kernel']:
        print(cmd, '--approx', approx)
        print(cmd, '--approx', approx, '--independent_outputs', '--single_output')

# TIMING commands lila
base_cmd = 'classification_image.py --config configs/bound_illustration_lila_timing.yaml'
for batch_size in [10, 20, 50, 100, 250, 500, 1000]:
    cmd = base_cmd + f' --marglik_batch_size {batch_size} --bound'
    for approx in ['full', 'kron', 'diag', 'kernel']:
        print(cmd, '--approx', approx)
        print(cmd, '--approx', approx, '--independent_outputs', '--single_output')
