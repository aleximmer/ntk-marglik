import logging
from time import time
import numpy as np
import torch
import wandb
from torch.nn.utils.convert_parameters import parameters_to_vector

from laplace import FullLaplace, KronLaplace, DiagLaplace, FunctionalLaplace, BlockDiagLaplace


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        try:
            from torch import cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except:
            pass


def get_laplace_approximation(structure):
    if structure == 'full':
        return FullLaplace
    elif structure == 'kron':
        return KronLaplace
    elif structure == 'diag':
        return DiagLaplace
    elif structure == 'blockdiag':
        return BlockDiagLaplace
    elif structure == 'kernel' or structure == 'kernel-stochastic':
        return FunctionalLaplace


def wandb_log_parameter_hist(model):
    for name, param in model.named_parameters():
        hist, edges = param.data.cpu().histogram(bins=64)
        wandb.log({f'params/{name}': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}, commit=False)


def wandb_log_parameter_norm(model):
    for name, param in model.named_parameters():
        avg_norm = (param.data.flatten() ** 2).sum().item() / np.prod(param.data.shape)
        wandb.log({f'params/{name}': avg_norm}, commit=False)


def wandb_log_invariance(augmenter):
    aug_params = np.abs(
        parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
    ).tolist()
    if len(aug_params) == 6:
        names = ['Tx', 'Ty', 'R', 'Sx', 'Sy', 'H']
    else:
        names = [f'aug_{i}' for i in range(6)]
    log = {f'invariances/{n}': p for n, p in zip(names, aug_params)}
    wandb.log(log, commit=False)


def wandb_log_prior(prior_prec, prior_structure, model):
    prior_prec = prior_prec.detach().cpu().numpy().tolist()
    if prior_structure == 'scalar':
        wandb.log({'hyperparams/prior_prec': prior_prec[0]}, commit=False)
    elif prior_structure == 'layerwise':
        log = {f'hyperparams/prior_prec_{n}': p for p, (n, _) in
               zip(prior_prec, model.named_parameters())}
        wandb.log(log, commit=False)
    elif prior_structure == 'diagonal':
        hist, edges = prior_prec.data.cpu().histogram(bins=64)
        log = {f'hyperparams/prior_prec': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}
        wandb.log(log, commit=False)


class Timer:
    def __init__(self, name, logger=False) -> None:
        self.logger = logger
        self.name = name
    def __enter__(self):
        self.start_time = time()
    def __exit__(self, *args, **kwargs):
        msg = f'{self.name} took {time() - self.start_time:.3f}s'
        print(msg)
        if self.logger:
            logging.info(msg)
