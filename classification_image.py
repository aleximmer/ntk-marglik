import logging
import torch
import wandb
from dotenv import load_dotenv
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature.asdl import AsdlGGN, AsdlEF
from laplace.curvature.augmented_asdl import AugAsdlGGN, AugAsdlEF

from ntkmarglik.marglik import marglik_optimization
from ntkmarglik.invariances import AffineLayer2d
from ntkmarglik.utils import get_laplace_approximation, set_seed
from ntkmarglik.models import MLP, LeNet, WideResNet, ResNet, MiniNet
from data_utils import (
    get_dataset, TensorDataLoader, SubsetTensorDataLoader, GroupedSubsetTensorDataLoader, 
    dataset_to_tensors, CIFAR_transform, MNIST_transform, ImageNet_transform
)


def main(
    seed, method, approx, curv, dataset, model, n_epochs, batch_size, marglik_batch_size, partial_batch_size, bound,
    subset_size, n_samples_aug, lr, lr_min, lr_hyp, lr_hyp_min, lr_aug, lr_aug_min, grouped_loader,
    prior_prec_init, n_epochs_burnin, marglik_frequency, n_hypersteps, n_hypersteps_prior, random_flip,
    device, download_data, data_root, independent_outputs, kron_jac, single_output,
    single_output_iid, temperature
):
    # dataset-specific static transforms (preprocessing)
    if 'mnist' in dataset:
        transform = MNIST_transform
    elif 'cifar' in dataset:
        transform = CIFAR_transform
    elif 'tiny' in dataset:
        transform = ImageNet_transform
    else:
        raise NotImplementedError(f'Transform for {dataset} unavailable.')

    train_dataset, test_dataset = get_dataset(dataset, data_root, download_data, transform)

    # dataset-specific number of classes
    if 'cifar100' in dataset:
        n_classes = 100
    elif 'tiny' in dataset:
        n_classes = 200
    else:
        n_classes = 10

    # Load data
    set_seed(seed)

    # Subset the data if subset_size is given.
    subset_size = len(train_dataset) if subset_size <= 0 else subset_size
    if subset_size < len(train_dataset):
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    else:
        subset_indices = None
    X_train, y_train = dataset_to_tensors(train_dataset, subset_indices, device)
    X_test, y_test = dataset_to_tensors(test_dataset, None, device)

    if method == 'lila':
        augmenter = AffineLayer2d(n_samples=n_samples_aug, init_value=0.0, random_flip=random_flip).to(device)
        augmenter_valid = augmenter_marglik = augmenter
        augmenter.rot_factor.requires_grad = True
    else:
        augmenter = augmenter_valid = augmenter_marglik = None
    optimize_aug = (method == 'lila')

    batch_size = subset_size if batch_size <= 0 else min(batch_size, subset_size)
    ml_batch_size = subset_size if marglik_batch_size <= 0 else min(marglik_batch_size, subset_size)
    pl_batch_size = subset_size if partial_batch_size <= 0 else min(partial_batch_size, subset_size)
    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=batch_size, shuffle=True, detach=True)
    valid_loader = TensorDataLoader(X_test, y_test, transform=augmenter_valid, batch_size=batch_size, detach=True)
    if not bound:
        stochastic_grad = False
        marglik_loader = TensorDataLoader(X_train, y_train, transform=augmenter_marglik, batch_size=ml_batch_size, shuffle=True, detach=True)
        partial_loader = TensorDataLoader(X_train, y_train, transform=augmenter_marglik, batch_size=pl_batch_size, shuffle=True, detach=False)
    else:  # use proposed lower bound estimators
        stochastic_grad = True
        data_factor = 1.0
        data_factor = len(X_train) / ml_batch_size
        DataLoaderCls = GroupedSubsetTensorDataLoader if grouped_loader else SubsetTensorDataLoader
        marglik_loader = DataLoaderCls(X_train, y_train, transform=augmenter_marglik, subset_size=ml_batch_size,
                                       detach=False, data_factor=data_factor)
        partial_loader = None

    # model
    optimizer = 'SGD'
    prior_structure = 'layerwise'
    if 'mnist' in dataset:
        if model == 'mlp':
            model = MLP(28*28, width=1000, depth=1, output_size=n_classes, activation='tanh', augmented=optimize_aug)
        elif model == 'mininet':
            optimizer = 'Adam'
            model = MiniNet(in_channels=1, n_out=10, augmented=optimize_aug)
        elif model == 'cnn':
            model = LeNet(in_channels=1, n_out=n_classes, activation='tanh', n_pixels=28, augmented=optimize_aug)
        else:
            raise ValueError('Unavailable model for (f)mnist')
    elif 'cifar' in dataset:
        if model == 'cnn':
            model = LeNet(in_channels=3, n_out=n_classes, activation='relu', n_pixels=32, augmented=optimize_aug)
        elif model == 'resnet':
            model = ResNet(depth=18, in_planes=16, num_classes=n_classes, augmented=optimize_aug)
        elif model == 'wrn':
            model = WideResNet(augmented=optimize_aug, num_classes=n_classes, depth=16, widen_factor=4)
        else:
            raise ValueError('Unavailable model for cifar')
    elif 'tiny' in dataset:
        assert model == 'resnet'
        model = ResNet(depth=50, in_planes=16, num_classes=n_classes, augmented=optimize_aug)

    model.to(device)

    laplace = get_laplace_approximation(approx)
    if optimize_aug:
        backend = AugAsdlGGN if curv in ['ggn', 'fisher'] else AugAsdlEF
    else:
        backend = AsdlGGN if curv in ['ggn', 'fisher'] else AsdlEF

    marglik_optimization(
        model, train_loader, marglik_loader, valid_loader, partial_loader, likelihood='classification',
        lr=lr, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min, lr_aug=lr_aug, n_epochs=n_epochs, 
        n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency, laplace=laplace,
        prior_structure=prior_structure, backend=backend, n_epochs_burnin=n_epochs_burnin,
        method=method, augmenter=augmenter_valid, lr_min=lr_min, scheduler='cos', optimizer=optimizer,
        n_hypersteps_prior=n_hypersteps_prior, temperature=temperature, lr_aug_min=lr_aug_min,
        prior_prec_init=prior_prec_init, stochastic_grad=stochastic_grad, use_wandb=True,
        independent=independent_outputs, kron_jac=kron_jac, single_output=single_output, 
        single_output_iid=single_output_iid
    )


if __name__ == '__main__':
    import sys
    import argparse
    from arg_utils import set_defaults_with_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--method', default='baseline', choices=['baseline', 'lila'])
    parser.add_argument('--approx', default='full', choices=['full', 'kron', 'blockdiag', 'diag', 'kernel'])
    parser.add_argument('--curv', default='ggn', choices=['ggn', 'ef'])
    parser.add_argument('--dataset', default='mnist', choices=[
        'mnist', 'mnist_r90', 'mnist_r180', 'translated_mnist', 'scaled_mnist', 'scaled_mnist2',
        'fmnist', 'fmnist_r90', 'fmnist_r180', 'translated_fmnist', 'scaled_fmnist', 'scaled_fmnist2',
        'cifar10', 'cifar10_r90', 'cifar10_r180', 'translated_cifar10', 'scaled_cifar10',
        'cifar100', 'cifar100_r90', 'cifar100_r180', 'translated_cifar100', 'scaled_cifar100',
        'tinyimagenet', 
    ])
    parser.add_argument('--model', default='mlp', choices=['mininet', 'mlp', 'cnn', 'resnet', 'wrn'])
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--independent_outputs', default=False, action=argparse.BooleanOptionalAction,
                        help='Independent outputs (only for Functional, use in conjunction with single_output)')
    parser.add_argument('--single_output', default=False, action=argparse.BooleanOptionalAction,
                        help='Sample only single output (output-wise partitioning)')
    parser.add_argument('--single_output_iid', default=False, action=argparse.BooleanOptionalAction,
                        help='Only applies when single_output=True. Sample single output iid instead of one output per batch.')
    parser.add_argument('--kron_jac', default=True, action=argparse.BooleanOptionalAction,
                        help='Use Kronecker (approximation) for Jacobians (where applicable). Always faster, sometimes an approx.')
    parser.add_argument('--subset_size', default=-1, type=int, help='Observations in generated data.')
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--marglik_frequency', default=1, type=int)
    parser.add_argument('--marglik_batch_size', default=-1, type=int, help='Used for fitting laplace.')
    parser.add_argument('--partial_batch_size', default=-1, type=int, help='Used for JVPs when necessary.')
    parser.add_argument('--grouped_loader', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bound', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use the lower bound estimators or aggregate full batch.')
    parser.add_argument('--n_hypersteps', default=1, help='Number of steps on every marglik estimate (partial grad accumulation)', type=int)
    parser.add_argument('--n_hypersteps_prior', default=1, help='Same as n_hypersteps but for the prior precision.', type=int)
    parser.add_argument('--n_samples_aug', default=31, type=int, help='Number of augmentation samples for lila.')
    parser.add_argument('--random_flip', default=False, action=argparse.BooleanOptionalAction,
                        help='Randomly flip input images (only for lila).')
    parser.add_argument('--prior_prec_init', default=1.0, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_min', default=0.1, type=float)
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.1, type=float)
    parser.add_argument('--lr_aug', default=0.005, type=float)
    parser.add_argument('--lr_aug_min', default=0.00001, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--download_data', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--config', nargs='+')
    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    args.pop('config')
    import uuid
    import copy
    bound_tag = 'lower' if args['bound'] else 'nobound'
    tags = [args['dataset'], args['model'], args['approx'], bound_tag]
    config = copy.deepcopy(args)
    config['map'] = (args['n_epochs_burnin'] > args['n_epochs'])
    if config['map']:  # MAP
        tags = [args['dataset'], args['model'], 'map']
    run_name = '-'.join(tags)
    if args['method'] == 'lila':
        run_name += '-lila'
    run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
    load_dotenv()
    wandb.init(project='ntk-marglik', config=config, name=run_name, tags=tags)
    main(**args)
