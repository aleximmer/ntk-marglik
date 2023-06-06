# TinyImageNet
base_cmd = 'classification_image.py --config configs/tiny.yaml'
seeds = [1, 2, 3]
for seed in seeds:
    cmd_parts = [base_cmd, f'--seed {seed}']
    cmd = ' '.join(cmd_parts)
    # MAP
    print(cmd, '--n_epochs_burnin 1000 --approx kron')
    # Marglik full batch updates
    print(cmd, '--approx kron')
    # full batch single output
    print(cmd, '--approx kron --single_output --independent_outputs')
    for approx in ['kron', 'kernel']:
        print(cmd, f'--approx {approx} --bound lower', '--independent_outputs --single_output')
        if approx == 'kron':
            print(cmd, f'--approx {approx} --bound lower')

base_cmd = 'classification_image.py --config configs/tiny_lila.yaml'
for seed in seeds:
    cmd_parts = [base_cmd, f'--seed {seed} --random_flip']
    cmd = ' '.join(cmd_parts)
    print(cmd, '--approx kernel')
    print(cmd, '--approx kron')
    # tiny_lila.yaml uses single output settings already
    print(cmd, '--approx kron --bound None --marglik_batch_size 60 --partial_batch_size 60 --n_hypersteps 100 --lr_aug 0.05 --lr_aug_min 0.005')
