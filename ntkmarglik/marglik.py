import logging
from time import time
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
import wandb

from laplace import KronLaplace, FunctionalLaplace
from laplace.curvature import AsdlGGN

from ntkmarglik.utils import (
    wandb_log_invariance, wandb_log_prior, wandb_log_parameter_norm
)

GB_FACTOR = 1024 ** 3


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec


def valid_performance(model, test_loader, likelihood, criterion, method, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            if method == 'lila':
                f = model(X).mean(dim=1)
            else:
                f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        else:
            perf += (f - y).square().sum() / N
        nll += criterion(f, y) / len(test_loader)
    return perf.item(), nll


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])


def marglik_optimization(model,
                         train_loader,
                         marglik_loader=None,
                         valid_loader=None,
                         partial_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-1,
                         lr_min=None,
                         optimizer='SGD',
                         scheduler='cos',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         n_hypersteps_prior=1,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_hyp_min=1e-1,
                         lr_aug=1e-2,
                         lr_aug_min=1e-2,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         independent=False,
                         single_output=False,
                         single_output_iid=False,
                         kron_jac=True,
                         method='baseline',
                         augmenter=None,
                         stochastic_grad=False,
                         use_wandb=False):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    marglik_loader : DataLoader
        pytorch data loader for fitting Laplace
    valid_loader : DataLoader
        pytorch data loader for validation
    partial_loader : DataLoader
        pytorch data loader for partial fitting for lila's grad accumulation
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on diff. hyperparameters when marglik is estimated
    n_hypersteps_prior : int
        how many steps to take on the prior when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    lr_hyp_min : float
        minimum learning rate, decayed to using cosine schedule
    lr_aug : float
        learning rate for augmentation parameters
    lr_aug_min : float
        minimum learning rate, decayed to using cosine schedule
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    independent : bool
        whether to use independent functional laplace
    single_output : bool
        whether to use single random output for functional laplace
    single_output_iid : bool
        whether to sample single output per sample iid (otherwise per batch)
    kron_jac : bool
        whether to use kron_jac in the backend
    method : augmentation strategy, one of ['baseline'] -> no change
        or ['lila'] -> change in protocol.
    augmenter : torch.nn.Module with differentiable parameter
    stochastic_grad : bool
        whether to use stochastic gradients of marginal likelihood
        usually would correspond to lower bound (unless small data)

    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    valid_perfs : list
    aug_history: list
        None for method == 'baseline'
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    if marglik_loader is None:
        marglik_loader = train_loader
    if partial_loader is None:
        partial_loader = marglik_loader
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    optimize_aug = augmenter is not None and parameters_to_vector(augmenter.parameters()).requires_grad
    backend_kwargs = dict(differentiable=(stochastic_grad and optimize_aug) or laplace is FunctionalLaplace,
                          kron_jac=kron_jac)
    la_kwargs = dict(sod=stochastic_grad, single_output=single_output, single_output_iid=single_output_iid)
    if laplace is FunctionalLaplace:
        la_kwargs['independent'] = independent
    if use_wandb:
        wandb.config.update(dict(n_params=P, n_param_groups=H, n_data=N))

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps_prior
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)
    if optimize_aug:
        logging.info('MARGLIK: optimize augmentation.')
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug)
        n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * (n_hypersteps if stochastic_grad else 1)
        aug_scheduler = CosineAnnealingLR(aug_optimizer, n_steps, eta_min=lr_aug_min)
        aug_history = [parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()]

    losses = list()
    valid_perfs = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_time_fwd = 0.0
        epoch_time_fit = 0.0

        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)

        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)

            # fit data
            time_fwd = time()
            if method == 'lila':
                f = model(X).mean(dim=1)
            else:
                f = model(X)
            epoch_time_fwd += time() - time_fwd # log total time fwd fit in epoch

            time_fit = time()
            theta = parameters_to_vector(model.parameters())
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()
            epoch_time_fit += time() - time_fit # log total time bwd fit in epoch

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        logging.info('MAP memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')
        logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf*100:.2f}%; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr,
                          'train/time_fwd': epoch_time_fwd, 'train/time_fit': epoch_time_fit})
        if use_wandb and ((epoch % 5) == 0):
            wandb_log_parameter_norm(model)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                val_perf, val_nll = valid_performance(model, valid_loader, likelihood, criterion, method, device)
                valid_perfs.append(val_perf)
                logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf*100:.2f}%; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            if use_wandb:
                wandb.log(epoch_log, step=epoch, commit=((epoch % 10) == 0))
            continue

        # optimizer hyperparameters by differentiating marglik
        time_hyper = time()

        # 1. fit laplace approximation
        torch.cuda.empty_cache()
        if optimize_aug:
            if stochastic_grad:  # differentiable
                marglik_loader.attach()
            else:  # jvp
                marglik_loader.detach()

        # first optimize prior precision jointly with direct marglik grad
        margliks_local = list()
        n_hyper = max(n_hypersteps_prior, n_hypersteps) if stochastic_grad else n_hypersteps_prior
        for i in range(n_hyper):
            if i == 0 or stochastic_grad:
                sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
                prior_prec = torch.exp(log_prior_prec)
                lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                              temperature=temperature, backend=backend, backend_kwargs=backend_kwargs,
                              **la_kwargs)
                lap.fit(marglik_loader)
            if i < n_hypersteps and optimize_aug and stochastic_grad:
                aug_optimizer.zero_grad()
            if i < n_hypersteps_prior:
                hyper_optimizer.zero_grad()
            if i < n_hypersteps_prior and not stochastic_grad:  # does not fit every it
                sigma_noise = None if likelihood == 'classification' else torch.exp(log_sigma_noise)
                prior_prec = torch.exp(log_prior_prec)
                marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N
            else:  # fit with updated hparams
                marglik = -lap.log_marginal_likelihood() / N
            marglik.backward()
            margliks_local.append(marglik.item())
            if i < n_hypersteps_prior:
                hyper_optimizer.step()
                hyper_scheduler.step()
            if i < n_hypersteps and optimize_aug and stochastic_grad:
                aug_optimizer.step()
                aug_scheduler.step()

        if stochastic_grad:
            marglik = np.mean(margliks_local)
        else:
            marglik = margliks_local[-1]

        if use_wandb:
            wandb_log_prior(torch.exp(log_prior_prec.detach()), prior_structure, model)
        if likelihood == 'regression':
            epoch_log['hyperparams/sigma_noise'] = torch.exp(log_sigma_noise.detach()).cpu().item()
        epoch_log['train/marglik'] = marglik
        logging.info('LA memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')

        # option 2: jvp (not direct_grad)
        torch.cuda.empty_cache()
        if optimize_aug and not stochastic_grad:  # accumulate gradient with JVP
            partial_loader.attach()
            aug_grad = torch.zeros_like(parameters_to_vector(augmenter.parameters()))
            lap.backend.differentiable = True
            if isinstance(lap, KronLaplace):
                # does the inversion internally
                hess_inv = lap.posterior_precision.jvp_logdet()
            else:
                hess_inv = lap.posterior_covariance.flatten()
            for i, (X, y) in zip(range(n_hypersteps), partial_loader):
                lap.loss, H_batch = lap._curv_closure(X, y, N)
                # curv closure creates gradient already, need to zero
                aug_optimizer.zero_grad()
                # compute grad wrt. neg. log-lik
                (- lap.log_likelihood).backward(inputs=list(augmenter.parameters()), retain_graph=True)
                # compute grad wrt. log det = 0.5 vec(P_inv) @ (grad-vec H)
                (0.5 * H_batch.flatten()).backward(gradient=hess_inv, inputs=list(augmenter.parameters()))
                aug_grad = (aug_grad + gradient_to_vector(augmenter.parameters()).data.clone())

            lap.backend.differentiable = False

            vector_to_gradient(aug_grad, augmenter.parameters())
            aug_optimizer.step()
            aug_scheduler.step()

        epoch_time_hyper = time() - time_hyper
        epoch_log.update({'train/time_hyper': epoch_time_hyper})

        if optimize_aug:
            aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())
            logging.info(f'Augmentation params epoch {epoch}: {aug_history[-1]}')
            if use_wandb:
                wandb_log_invariance(augmenter)

        logging.info('LA memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')

        margliks.append(marglik)
        del lap
        if use_wandb:
            if optimize_aug:
                epoch_log['train/lr_aug'] = aug_scheduler.get_last_lr()[0]
            epoch_log['train/lr_hyp'] = hyper_scheduler.get_last_lr()[0]
            wandb.log(epoch_log, step=epoch, commit=((epoch % 10) == 0))

        # early stopping on marginal likelihood
        logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.5f}, prec: {prior_prec.detach().mean().item():.2f}.')

    sigma_noise = 1 if sigma_noise is None else sigma_noise
    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend, backend_kwargs=backend_kwargs,
                  **la_kwargs)
    lap.fit(marglik_loader.detach())
    if optimize_aug:
        return lap, model, margliks, valid_perfs, aug_history
    return lap, model, margliks, valid_perfs, None
