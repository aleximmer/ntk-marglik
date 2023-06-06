# CIFAR10 and CIFAR100
base_cmd = 'classification_image.py'
configs = ['--config configs/cifar.yaml --dataset cifar10', '--config configs/cifar.yaml --dataset cifar100']
seeds = [1, 2, 3, 4, 5]
model = 'wrn'
for config in configs:
    for seed in seeds:
        cmd_parts = [base_cmd, config, f'--model {model} --seed {seed} --batch_size 128 --lr_min 1e-8 --n_epochs 300']
        cmd = ' '.join(cmd_parts)
        # MAP
        print(cmd, '--n_epochs_burnin 1000 --approx kron')
        # Marglik full batch updates
        print(cmd, '--approx kron')
        print(cmd, '--approx kron --single_output --independent_outputs')
        print(cmd, '--approx kron --single_output --independent_outputs --single_output_iid')
        for approx in ['kron', 'kernel']:
            print(cmd, f'--approx {approx} --bound', '--independent_outputs --single_output')
            print(cmd, f'--approx {approx} --bound', '--independent_outputs --single_output --single_output_iid')
            print(cmd, f'--approx {approx} --bound', '--independent_outputs --single_output --grouped_loader')
            if approx == 'kron':
                print(cmd, f'--approx {approx} --bound')
                print(cmd, f'--approx {approx} --bound', '--grouped_loader')
