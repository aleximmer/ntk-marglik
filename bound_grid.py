import logging
import pandas as pd
import numpy as np
from scipy.stats import sem
import torch
from torchvision import transforms
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature.asdl import AsdlGGN
from laplace.curvature.augmented_asdl import AugAsdlGGN
from laplace import FunctionalLaplace, BlockDiagLaplace, KronLaplace, DiagLaplace, FullLaplace

from ntkmarglik.marglik import marglik_optimization
from ntkmarglik.invariances import AffineLayer2d
from ntkmarglik.utils import set_seed
from ntkmarglik.models import MiniNet
from data_utils.utils import TensorDataLoader, SubsetTensorDataLoader, dataset_to_tensors, GroupedSubsetTensorDataLoader
from classification_image import get_dataset

INVARIANCE = 'invariance'
PRIOR = 'prior'
DATA_ROOT = '/is/cluster/fast/aimmer/data'


def get_marglik_loader(x, y, batch_size, augmenter, grouped_loader):
    data_factor = len(x) / batch_size
    DataLoaderCls = GroupedSubsetTensorDataLoader if grouped_loader else SubsetTensorDataLoader
    marglik_loader = DataLoaderCls(x, y, transform=augmenter, subset_size=batch_size,
                                   detach=False, data_factor=data_factor)
    return marglik_loader


def main(setting, approximation, single_output, grouped_loader, stochastic):
    device = 'cuda'
    subset_size = 1000
    batch_size = 250
    marglik_batch_size = 1000
    lr, lr_min, lr_hyp, lr_hyp_min, lr_aug, lr_aug_min = 1e-3, 1e-4, 1e-1, 1e-2, 0.05, 0.005
    n_epochs = 100
    n_epochs_burnin = 10
    if setting == PRIOR:
        dataset = 'mnist'
        marglik_frequency = 5
        n_hypersteps_prior = 10
        n_hypersteps = 1
    else:
        dataset = 'mnist_r180'
        marglik_frequency = 1
        n_hypersteps_prior = 2
        n_hypersteps = 2

    ####### Quickly train with NTK Laplace
    transform = transforms.ToTensor()
    train_dataset, _ = get_dataset(dataset, DATA_ROOT, False, transform)
    set_seed(711)
    subset_size = len(train_dataset) if subset_size <= 0 else subset_size
    if subset_size < len(train_dataset):
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    else:
        subset_indices = None
    X_train, y_train = dataset_to_tensors(train_dataset, subset_indices, device)
    if setting == INVARIANCE:
        augmenter = AffineLayer2d(n_samples=30).to(device)
        augmenter_valid = augmenter
        augmenter.rot_factor.requires_grad = True
    else:
        augmenter = augmenter_valid = None

    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=batch_size, shuffle=True, detach=True)
    marglik_loader = get_marglik_loader(X_train, y_train, marglik_batch_size, augmenter, grouped_loader=False)
    partial_loader = None
    stochastic_grad = True

    optimizer = 'Adam'
    prior_structure = 'scalar'
    model = MiniNet(in_channels=1, n_out=10, augmented=(setting == INVARIANCE)).to(device)
    backend = AugAsdlGGN if setting == INVARIANCE else AsdlGGN
    method = 'lila' if setting == INVARIANCE else 'baseline'

    la, model, margliks, valid_perfs, aug_history = marglik_optimization(
        model, train_loader, marglik_loader, None, partial_loader, likelihood='classification',
        lr=lr, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min, lr_aug=lr_aug, n_epochs=n_epochs,
        n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency, laplace=FunctionalLaplace,
        prior_structure=prior_structure, backend=backend, n_epochs_burnin=n_epochs_burnin,
        method=method, augmenter=augmenter_valid, lr_min=lr_min, scheduler='cos', optimizer=optimizer,
        n_hypersteps_prior=n_hypersteps_prior, temperature=1.0, lr_aug_min=lr_aug_min,
        prior_prec_init=1.0, stochastic_grad=stochastic_grad, use_wandb=False,
        independent=False, kron_jac=False, single_output=False
    )
    ####### Finished training.

    ####### Assess bound at converged setting
    backend_kwargs = dict(differentiable=False, kron_jac=False)
    la_kwargs = dict(sod=True, single_output=single_output)
    if approximation == 'kernel' and single_output:
        la_kwargs['independent'] = True

    if stochastic:
        batch_sizes = [10, 20, 50, 100, 250, 500, 1000]
    else:
        # for parametric no sod bounds
        batch_sizes = [1000]

    grid = np.logspace(-4, 4, 100) if setting == PRIOR else np.linspace(0, np.pi, 100)
    result_frame = pd.DataFrame(index=batch_sizes, columns=grid)
    result_frame_sem = pd.DataFrame(index=batch_sizes, columns=grid)
    for batch_size in batch_sizes:
        for hparam in grid:
            set_seed(711)
            marglik_loader = get_marglik_loader(X_train, y_train, batch_size, augmenter, grouped_loader)
            marglik_loader = marglik_loader.detach()
            if setting == INVARIANCE:
                augmenter.rot_factor.requires_grad = False
                augmenter.rot_factor.data[2] = float(hparam)
                prior_precision = la.prior_precision
            else:
                prior_precision = float(hparam)

            margliks = list()
            n_reps = int(subset_size / batch_size)
            for rep in range(n_reps):
                if approximation == 'kernel':
                    lap_cls = FunctionalLaplace
                elif approximation == 'full':
                    lap_cls = FullLaplace
                elif approximation == 'blockdiag':
                    lap_cls = BlockDiagLaplace
                elif approximation == 'kron':
                    lap_cls = KronLaplace
                elif approximation == 'diag':
                    lap_cls = DiagLaplace
                lap = lap_cls(model, 'classification', prior_precision=prior_precision,
                              backend=backend, backend_kwargs=backend_kwargs, **la_kwargs)
                lap.fit(marglik_loader)
                marglik = lap.log_marginal_likelihood().item() / subset_size
                margliks.append(marglik)
            result_frame.loc[batch_size, hparam] = np.mean(margliks)
            result_frame_sem.loc[batch_size, hparam] = sem(margliks)
            print(setting, batch_size, hparam, np.mean(margliks), np.nan_to_num(sem(margliks)))

    str_id = f'{setting}_{approximation}_so={single_output}_grouped={grouped_loader}_sto={stochastic}'
    result_frame.to_csv(f'results_grid/grid_bound_{str_id}.csv')
    result_frame_sem.to_csv(f'results_grid/grid_bound_sem_{str_id}.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=[PRIOR, INVARIANCE])
    parser.add_argument('--approximation', choices=['kernel', 'full', 'blockdiag', 'kron', 'diag'])
    parser.add_argument('--single_output', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--grouped_loader', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--stochastic', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args.setting, args.approximation, args.single_output, args.grouped_loader, args.stochastic)
