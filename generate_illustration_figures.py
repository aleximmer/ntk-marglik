import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes, axes
from scipy.interpolate import interp1d

import wandb


def get_runs(query):
    api = wandb.Api()
    return api.runs('aleximmer/ntk-marglik', query) 


def get_frame(runs, group_keys):
    df_marglik = pd.DataFrame(columns=group_keys)
    df_nll = pd.DataFrame(columns=group_keys)
    for i, run in enumerate(runs):
        if run.state != 'finished':
            continue
        for k in group_keys:
            df_marglik.loc[i, k] = df_nll.loc[i, k] = run.config[k]
        df_run = run.history(keys=['train/marglik', 'valid/nll'])
        try:
            steps = df_run._step.values.astype(int)
        except:
            print(run.config, 'failed')
            print(df_run.head())
            continue
        df_marglik.loc[i, df_run._step.values.astype(int)] = df_run['train/marglik'].values
        df_nll.loc[i, df_run._step.values.astype(int)] = df_run['valid/nll'].values
    steps = pd.DataFrame(index=steps)
    return steps, df_marglik, df_nll


def get_runtime_frame(runs, group_keys):
    df = pd.DataFrame(columns=group_keys + ['runtime'])
    for i, run in enumerate(runs):
        df.loc[i, group_keys] = [run.config[k] for k in group_keys]
        df_run = run.history(keys=['train/time_hyper'])
        df.loc[i, 'runtime'] = df_run.iloc[-3:].mean()['train/time_hyper']
    return df
    

def get_map_baseline():
    query = {'$and': [{'config.model': 'mininet'}, {'config.map': True}]}
    runs = get_runs(query)
    df_nll = pd.DataFrame(columns=['seed'])
    for i, run in enumerate(runs):
        df_nll.loc[i, 'seed'] = run.config['seed']
        # df_nll.loc[i, 'nll'] = run.summary['valid/nll']
        df_run = run.history(keys=['valid/nll'])
        steps = df_run._step.values.astype(int)
        df_nll.loc[i, steps] = df_run['valid/nll'].values
    steps = pd.DataFrame(index=steps)
    return steps, df_nll.drop('seed', axis=1)

    
def load_with_cache(base_name, query, group_keys):
    try:
        print('Trying to load locally')
        steps = pd.read_csv(f'results/{base_name}_steps.csv', index_col=0)
        marglik = pd.read_csv(f'results/{base_name}_marglik.csv', index_col=0)
        nll = pd.read_csv(f'results/{base_name}_nll.csv', index_col=0)
    except:
        print('Failed local loading. Loading from wandb.')
        steps, marglik, nll = get_frame(get_runs(query), group_keys)
        steps.to_csv(f'results/{base_name}_steps.csv')
        marglik.to_csv(f'results/{base_name}_marglik.csv')
        nll.to_csv(f'results/{base_name}_nll.csv')
    steps = steps.index.values
    return steps, marglik, nll

    
def parametric_bound_figure(method='baseline'):
    base_name = 'parametric_bound' + ('' if method == 'baseline' else '_lila')
    query = {
        '$and': [{'config.model': 'mininet'}, {'config.bound': 'lower'}, 
                 {'config.marglik_batch_size': 1000}, {'config.single_output': False},
                 {'config.grouped_loader': False}, {'config.approx': {'$ne': 'kernel'}},
                 {'config.method': method}, {'config.n_epochs': 500}]
    }
    steps, marglik, nll = load_with_cache(base_name, query, ['approx', 'seed'])
    seeds = len(set(marglik['seed']))
    marglik = marglik.drop('seed', axis=1)
    nll = nll.drop('seed', axis=1)
    marglik_group = marglik.groupby(by=['approx'])
    nll_group = nll.groupby(by=['approx'])
    marglik_m, marglik_ste = marglik_group.mean(), marglik_group.std() / np.sqrt(seeds)
    nll_m, nll_ste = nll_group.mean(), nll_group.std() / np.sqrt(seeds)

    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(height_to_width_ratio=0.38)}):
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
        axs[0].grid()
        axs[1].grid()
        axs[0].set_xlim([10, 500])
        axs[1].set_xlim([10, 500])
        if method == 'baseline':
            axs[0].set_ylim([-3, -0.48])
            axs[1].set_ylim([-0.8, -0.18])
        else:
            axs[0].set_ylim([-4, -0.7])
            axs[1].set_ylim([-2.3, -0.3])
        axs[0].set_ylabel('$\log q(\mathcal{D} | \mathbf{h})$')
        axs[1].set_ylabel('test log likelihood')
        axs[0].set_xlabel('steps')
        axs[1].set_xlabel('steps')

        approxs = ['full', 'blockdiag', 'kron', 'diag']
        blue = blue = mpl.cm.get_cmap('Blues')(1.0)
        colors = [blue, 'tab:red', 'tab:orange', 'tab:purple']

        for i, approx in enumerate(approxs):
            color = colors[i]
            m, ste = -marglik_m.loc[approx].rolling(5).mean(), marglik_ste.loc[approx].rolling(5).mean()
            axs[0].plot(steps, m, label=approx, c=color, alpha=0.9)
            axs[0].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

            m, ste = -nll_m.loc[approx].rolling(5).mean(), nll_ste.loc[approx].rolling(5).mean()
            axs[1].plot(steps, m, label=approx, c=color, alpha=0.9)
            axs[1].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

        axs[1].legend()
        # plt.savefig(f'figures/{base_name}.pdf')
        plt.show()


def bound_figure(method, base_name, approx, cmap):
    base_name = f'{base_name}_bound' + ('' if method == 'baseline' else '_lila')
    query = {
        '$and': [{'config.model': 'mininet'}, {'config.approx': approx},
                 {'config.single_output': False}, {'config.grouped_loader': False},
                 {'config.bound': 'lower'}, {'config.method': method},
                 {'config.n_epochs': 500}]
    }
    steps, marglik, nll = load_with_cache(base_name, query, ['marglik_batch_size', 'seed'])
    seeds = len(set(marglik['seed']))
    marglik = marglik.drop('seed', axis=1)
    nll = nll.drop('seed', axis=1)
    marglik_group = marglik.groupby(by=['marglik_batch_size'])
    nll_group = nll.groupby(by=['marglik_batch_size'])
    marglik_m, marglik_ste = marglik_group.mean(), marglik_group.std() / np.sqrt(seeds)
    nll_m, nll_ste = nll_group.mean(), nll_group.std() / np.sqrt(seeds)

    cmap = plt.cm.get_cmap(cmap)
    batch_sizes = sorted(list(set(marglik['marglik_batch_size'])))

    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(height_to_width_ratio=0.38)}):
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, 
                                gridspec_kw={'width_ratios': [10, 10, 1]})
        axs[0].grid()
        axs[1].grid()
        axs[0].set_xlim([10, 500])
        axs[1].set_xlim([10, 500])
        if method == 'baseline':
            axs[0].set_ylim([-3, -0.48])
            axs[1].set_ylim([-0.8, -0.18])
        else:
            axs[0].set_ylim([-4, -0.7])
            axs[1].set_ylim([-2.3, -0.3])
        axs[0].set_ylabel('$\log q(\mathcal{D} | \mathbf{h})$')
        axs[1].set_ylabel('test log likelihood')
        axs[0].set_xlabel('steps')
        axs[1].set_xlabel('steps')
        min_bs, max_bs = min(batch_sizes), max(batch_sizes)
        norm = mpl.colors.LogNorm(vmin=min_bs/10, vmax=max_bs)

        for bs in batch_sizes:
            color = cmap(norm(bs))
            m, ste = -marglik_m.loc[bs].rolling(5).mean(), marglik_ste.loc[bs].rolling(5).mean()
            axs[0].plot(steps, m, label=bs, c=color)
            axs[0].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

            m, ste = -nll_m.loc[bs].rolling(5).mean(), nll_ste.loc[bs].rolling(5).mean()
            axs[1].plot(steps, m, label=bs, c=color)
            axs[1].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs[2])
        axs[2].set_ylim([10, 1000])
        cbar.minorticks_off()
        cbar.set_label('indices $|\mathcal{B}_m|$')
        cbar.set_ticks(batch_sizes)
        C = 10
        cbar.set_ticklabels([str(e*C) for e in batch_sizes])
        # plt.savefig(f'figures/{base_name}.pdf')
        plt.show()


def classwise_figure(method, base_name, approx, cmap):
    base_name = f'{base_name}_classwise' + ('' if method == 'baseline' else '_lila')
    query = {
        '$and': [{'config.model': 'mininet'}, {'config.approx': approx},
                 {'config.bound': 'lower'}, {'config.method': method},
                 {'config.grouped_loader': False}, {'config.n_epochs': 500},
                 {'config.single_output': True}]
    }
    steps, marglik, nll = load_with_cache(base_name, query, ['marglik_batch_size', 'seed'])
    seeds = len(set(marglik['seed']))
    marglik = marglik.drop('seed', axis=1)
    nll = nll.drop('seed', axis=1)
    marglik_group = marglik.groupby(by=['marglik_batch_size'])
    nll_group = nll.groupby(by=['marglik_batch_size'])
    marglik_m, marglik_ste = marglik_group.mean(), marglik_group.std() / np.sqrt(seeds)
    nll_m, nll_ste = nll_group.mean(), nll_group.std() / np.sqrt(seeds)
    cmap = plt.cm.get_cmap(cmap)

    batch_sizes = sorted(list(set(marglik['marglik_batch_size'])))

    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(height_to_width_ratio=0.38)}):
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, 
                                gridspec_kw={'width_ratios': [10, 10, 1]})
        axs[0].grid()
        axs[1].grid()
        axs[0].set_xlim([10, 500])
        axs[1].set_xlim([10, 500])
        if method == 'baseline':
            axs[0].set_ylim([-3, -0.48])
            axs[1].set_ylim([-0.8, -0.18])
        else:
            axs[0].set_ylim([-4, -0.7])
            axs[1].set_ylim([-2.3, -0.3])
        axs[0].set_ylabel('$\log q(\mathcal{D} | \mathbf{h})$')
        axs[1].set_ylabel('test log likelihood')
        axs[0].set_xlabel('steps')
        axs[1].set_xlabel('steps')
        min_bs, max_bs = min(batch_sizes), max(batch_sizes)
        norm = mpl.colors.LogNorm(vmin=min_bs/10, vmax=max_bs)

        for bs in batch_sizes:
            color = cmap(norm(bs))
            m, ste = -marglik_m.loc[bs].rolling(5).mean(), marglik_ste.loc[bs].rolling(5).mean()
            axs[0].plot(steps, m, label=bs, c=color)
            axs[0].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

            m, ste = -nll_m.loc[bs].rolling(5).mean(), nll_ste.loc[bs].rolling(5).mean()
            axs[1].plot(steps, m, label=bs, c=color)
            axs[1].fill_between(steps, m-ste, m+ste, alpha=0.3, color=color)

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs[2])
        axs[2].set_ylim([10, 1000])
        cbar.minorticks_off()
        cbar.set_label('indices $|\mathcal{B}_m|$')
        cbar.set_ticks(batch_sizes)
        cbar.set_ticklabels([str(e) for e in batch_sizes])
        # plt.savefig(f'figures/{base_name}.pdf')
        plt.show()

        
def pareto_figure(method):
    query = {
        '$and': [{'config.model': 'mininet', 'config.bound': 'lower', 'config.grouped_loader': False, 
                  'config.method': method, 'config.n_epochs': 500, 'config.approx': {'$ne': 'blockdiag'}}]
    }
    # runs = get_runs(query_nobound_lila)
    query_time = {
        '$and': [{'config.model': 'mininet', 'config.bound': 'lower', 'config.grouped_loader': False, 
                  'config.method': method, 'config.n_epochs': 5}]
    }
    group_keys = ['marglik_batch_size', 'single_output', 'approx']
    _, df_marglik, df_nll = load_with_cache('pareto_' + method, query, group_keys)
    margliks = df_marglik.groupby(['marglik_batch_size', 'single_output', 'approx']).mean().iloc[:, -1]
    nlls = df_nll.groupby(['marglik_batch_size', 'single_output', 'approx']).mean().iloc[:, -1]
    df_time = get_runtime_frame(get_runs(query_time), group_keys)
    df = df_time.groupby(group_keys).mean()
    df.loc[margliks.index, 'marglik'] = margliks
    df.loc[margliks.index, 'nll'] = nlls
    df = df.reset_index()
    df.loc[~df.single_output, 'marglik_batch_size'] *= 10

    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(),
                        **{'figure.figsize': (2.75, 2.0)}}):
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, 
                                gridspec_kw={'width_ratios': [20, 1]})
        scax, cax = axs
        scax.grid()
        
        batch_sizes = sorted(list(set(df['marglik_batch_size'])))
        batch_sizes.remove(200)
        
        # marker types
        approx_to_marker = {
            'full': 's', 'kernel': 'd', 'kron': 'P', 'diag': '^'
        }
        df['marker'] = [approx_to_marker[e] for e in df.approx]
        df['line'] = None
        
        # labels
        approxs = ['full', 'kernel', 'kron', 'diag']
        labels = ['Full $\\textsc{ggn}$', '$\\textsc{ntk}$', '$\\textsc{kfac}$', 'Diag $\\textsc{ggn}$']
        mapper = {app: lbl for app, lbl in zip(approxs, labels)}
        df['label'] = [mapper[e] for e in df.approx]
        
        
        # colors
        cmap = plt.cm.get_cmap('inferno_r')
        min_bs, max_bs = min(df.marglik_batch_size), max(df.marglik_batch_size)
        norm = mpl.colors.LogNorm(vmin=min_bs, vmax=max_bs)
        
        for i, row in df.iterrows():
            color = cmap(norm(row.marglik_batch_size))
            line = scax.scatter(row.runtime, row.marglik, marker=row.marker, s=30 if row.approx == 'full' else 40,
                            facecolor=color, label=row.label, linewidth=0.1, alpha=0.9, color='black')
            df.loc[i, 'line'] = line
        
        handles = list()
        base_mask = (df.marglik_batch_size == 10000)
        for approx in approxs:
            handles.append(df.loc[(df.approx == approx) & base_mask, 'line'].values[0])
        scax.legend(handles, labels)
            
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cax.set_ylim([10, 10000])
        cbar.minorticks_off()
        cbar.set_label('batch size')
        cbar.set_ticks(batch_sizes)
        cbar.set_ticklabels([str(e) for e in batch_sizes])

        if method == 'lila':
            scax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
            scax.set_yticks([0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
            scax.set_ylim([0.7, 2.57])
            scax.set_xlim([-0.25, 3.25])
            pareto_line = np.array([[0.05, 2.7], [0.072, 2.4], [0.08, 1.6], [0.1, 1.4], [0.2, 1.2],
                                    [0.25, 1.02], [0.8, 0.95], [1.8, 0.82], [3.0, 0.815], [3.2, 0.812]])
            xs = np.linspace(0.05, 3.2, 100)
            f = interp1d(pareto_line[:, 0], pareto_line[:, 1], kind='slinear')
            scax.plot(xs-0.05, f(xs)-0.02, color='black', ls='--')
            scax.text(0.25, 0.8, 'Pareto', rotation=-10)
        else:
            scax.set_xscale('log')
            base_pareto_line = np.array([[0.0139, 2.9], [0.0145, 2.3], [0.022, 1.25], [0.028, 1.0], 
                                         [0.045, 0.71], [0.35, 0.69], [1.0, 0.55], [1.2, 0.549]])
            f = interp1d(base_pareto_line[:, 0], base_pareto_line[:, 1], kind='slinear')
            xs = np.linspace(0.0139, 1.2, 100)
            scax.plot(xs*0.9, f(xs)*0.9, color='black', ls='--')
            scax.set_xlim([9e-3, 1.7])
            scax.set_ylim([0.4, 2.53])
        
        scax.set_ylabel('Negative Log Marginal Likelihood')
        scax.set_xlabel('Runtime (s)')
        # plt.savefig(f'figures/pareto_{method}.pdf')
        plt.show()


def grid_bound_stochastic_figure(approximation, cmap, single_output=False, grouped_loader=False):
    str_id = f'{approximation}_so={single_output}_grouped={grouped_loader}_sto=True'
    df_prior = pd.read_csv(f'results_grid/grid_bound_prior_{str_id}.csv', index_col=0).astype(float)
    df_inv = pd.read_csv(f'results_grid/grid_bound_invariance_{str_id}.csv', index_col=0).astype(float)
    df_prior_sem = pd.read_csv(f'results_grid/grid_bound_sem_prior_{str_id}.csv', index_col=0).astype(float)
    df_inv_sem = pd.read_csv(f'results_grid/grid_bound_sem_invariance_{str_id}.csv', index_col=0).astype(float)
    cmap = plt.cm.get_cmap(cmap)
    batch_sizes = list(df_prior.index)
    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(height_to_width_ratio=0.38)}):
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, 
                                gridspec_kw={'width_ratios': [10, 10, 1]})
        axs[0].grid()
        axs[1].grid()
        # axs[0].set_title('Prior MNIST')
        # axs[1].set_title('Invar. rotated MNIST')
        axs[0].set_xscale('log')
        axs[0].set_xlim([1e-2, 5e2])
        ymin = -6 if approximation == 'kernel' else -7
        axs[0].set_yticks([-2, -4, -6])
        axs[1].set_yticks([-2, -4, -6])
        axs[0].set_ylim([ymin, -0.5])
        axs[1].set_ylim([ymin, -1.1])
        axs[0].set_ylabel('$\log q_{\\tilde{\mathbf{w}}}(\mathcal{D} | \mathbf{h})$')
        axs[0].set_xlabel('prior precision')
        axs[1].set_xlabel('rotational invariance')
        axs[1].set_xticks([0, np.pi/2, np.pi])
        axs[1].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        axs[1].set_xlim([0, np.pi])
        min_bs, max_bs = min(batch_sizes), max(batch_sizes)
        norm = mpl.colors.LogNorm(vmin=min_bs/10, vmax=max_bs)

        for bs in batch_sizes:
            xs = df_prior.columns.astype(float)
            color = cmap(norm(bs))
            ys = df_prior.loc[bs]
            yerrs = df_prior_sem.loc[bs]
            mask = (~ np.isnan(ys))
            axs[0].plot(xs[mask], ys[mask], label=bs, c=color)
            axs[0].fill_between(xs[mask], ys[mask] - yerrs[mask], ys[mask] + yerrs[mask], alpha=0.2, color=color)
            xs = df_inv.columns.astype(float)
            ys = df_inv.loc[bs]
            yerrs = df_inv_sem.loc[bs]
            mask = (~ np.isnan(ys))
            axs[1].plot(xs[mask], ys[mask], label=bs, c=color)
            axs[1].fill_between(xs[mask], ys[mask] - yerrs[mask], ys[mask] + yerrs[mask], alpha=0.2, color=color)

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs[2])
        axs[2].set_ylim([10, 1000])
        cbar.minorticks_off()
        cbar.set_label('indices $|\mathcal{B}_m|$')
        cbar.set_ticks(batch_sizes)
        C = 1 if single_output else 10 
        cbar.set_ticklabels([str(e*C) for e in batch_sizes])
        identifier = f'{approximation}_so={single_output}_grouped={grouped_loader}'
        # plt.savefig(f'figures/grid_bound_{identifier}.pdf')
        plt.show()


def grid_bound_parametric_figure():
    approxs = ['full', 'blockdiag', 'kron', 'diag']
    blue = blue = mpl.cm.get_cmap('Blues')(1.0)
    colors = [blue, 'tab:red', 'tab:orange', 'tab:purple']
    results = dict()
    for approximation in approxs:
        str_id = f'{approximation}_so=False_grouped=False_sto=False'
        df_prior = pd.read_csv(f'results_grid/grid_bound_prior_{str_id}.csv', index_col=0).astype(float)
        df_inv = pd.read_csv(f'results_grid/grid_bound_invariance_{str_id}.csv', index_col=0).astype(float)
        df_prior_sem = pd.read_csv(f'results_grid/grid_bound_sem_prior_{str_id}.csv', index_col=0).astype(float)
        df_inv_sem = pd.read_csv(f'results_grid/grid_bound_sem_invariance_{str_id}.csv', index_col=0).astype(float)
        results[approximation] = dict(prior=dict(mean=df_prior, sem=df_prior_sem), inv=dict(mean=df_inv, sem=df_inv_sem))
    
    with plt.rc_context({**bundles.icml2022(column='half'), 
                        **axes.lines(), 
                        **figsizes.icml2022_half(height_to_width_ratio=0.38)}):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 3], wspace=0.00)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1_top = fig.add_subplot(gs[0, 1])
        ax1_bottom = fig.add_subplot(gs[1, 1])
        axs = [ax0, ax1_top, ax1_bottom]
        ax1_top.set_xticklabels([])
        ax1_top.set_xticks([])
        ax1_top.set_xlabel('')
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[0].set_xscale('log')
        axs[0].set_xlim([1e-2, 1e3])
        axs[0].set_ylim([-6, -0.5])
        axs[0].set_ylabel('$\log q_{\\tilde{\mathbf{w}}}(\mathcal{D} | \mathbf{h})$')
        axs[0].set_xlabel('prior precision')
        axs[2].set_xlabel('rotational invariance')
        axs[2].set_xticks([0, np.pi/2, np.pi])
        axs[2].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        axs[2].set_xlim([0, np.pi])
        axs[1].set_xlim([0, np.pi])
        axs[1].spines.bottom.set_visible(False)
        axs[1].set_xticks([np.pi/2])
        axs[1].tick_params(labelbottom=False)  # don't put tick labels at the top
        axs[2].spines.top.set_visible(False)
        d = .33  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1_top.plot([0, 1], [0, 0], transform=ax1_top.transAxes, **kwargs)
        ax1_bottom.plot([0, 1], [1, 1], transform=ax1_bottom.transAxes, **kwargs)

        for approx, color in zip(approxs, colors):
            df_prior = results[approx]['prior']['mean']
            df_inv = results[approx]['inv']['mean']
            df_prior_sem = results[approx]['prior']['sem']
            df_inv_sem = results[approx]['inv']['sem']
            xs = df_prior.columns.astype(float)
            ys = df_prior.loc[1000].values.astype(float)
            yerr = df_prior_sem.loc[1000].values.astype(float)
            axs[0].plot(xs, ys, label=approx, c=color)
            axs[0].fill_between(xs, ys - yerr, ys + yerr, alpha=0.2, color=color)
            xs = df_inv.columns.astype(float)
            ys = df_inv.loc[1000].values.astype(float)
            yerr = df_inv_sem.loc[1000].values.astype(float)
            if approx == 'diag':
                axs[2].plot(xs, ys, label=approx, c=color)
                axs[2].fill_between(xs, ys - yerr, ys + yerr, alpha=0.2, color=color)
            else:
                axs[1].plot(xs, ys, label=approx, c=color)
                axs[1].fill_between(xs, ys - yerr, ys + yerr, alpha=0.2, color=color)
        axs[0].legend(loc='lower left')
        # plt.savefig(f'figures/grid_bound_parametric.pdf')
        plt.show()


if __name__ == '__main__':
    grid_bound_parametric_figure()
    grid_bound_stochastic_figure('kernel', 'Blues')
    grid_bound_stochastic_figure('kernel', 'Blues', single_output=True)
    grid_bound_stochastic_figure('kernel', 'Blues', grouped_loader=True)
    grid_bound_stochastic_figure('kernel', 'Blues', single_output=True, grouped_loader=True)
    grid_bound_stochastic_figure('blockdiag', 'Reds')
    grid_bound_stochastic_figure('kron', 'Oranges')
    pareto_figure('lila')
    pareto_figure('baseline')

    parametric_bound_figure('baseline')
    bound_figure('baseline', 'kernel', 'kernel', 'Blues')
    bound_figure('baseline', 'doubly_full', 'full', 'Blues')
    bound_figure('baseline', 'doubly_block', 'blockdiag', 'Reds')
    bound_figure('baseline', 'doubly_kron', 'kron', 'Oranges')
    bound_figure('baseline', 'doubly_diag', 'diag', 'Purples')
    classwise_figure('baseline', 'kernel', 'kernel', 'Blues')
    classwise_figure('baseline', 'kron', 'kron', 'Oranges')
    classwise_figure('baseline', 'blockdiag', 'blockdiag', 'Reds')

    parametric_bound_figure('lila')
    bound_figure('lila', 'kernel', 'kernel', 'Blues')
    bound_figure('lila', 'doubly_full', 'full', 'Blues')
    bound_figure('lila', 'doubly_block', 'blockdiag', 'Reds')
    bound_figure('lila', 'doubly_kron', 'kron', 'Oranges')
    bound_figure('lila', 'doubly_diag', 'diag', 'Purples')
    classwise_figure('lila', 'kernel', 'kernel', 'Blues')
    classwise_figure('lila', 'kron', 'kron', 'Oranges')
    classwise_figure('lila', 'blockdiag', 'blockdiag', 'Reds')
    
    # improvement of using grouping
    for method in ['baseline', 'lila']:
        print('*' * 10 + method + '*' * 10)
        for approx in ['full', 'kernel', 'kron', 'diag']:
            base_name = f'{approx}_partition' + ('' if method == 'baseline' else '_lila')
            query = {
                '$and': [{'config.model': 'mininet'}, {'config.approx': approx},
                         {'config.bound': 'lower'}, {'config.method': method},
                         {'config.n_epochs': 500}]
            }
            steps, marglik, nll = load_with_cache(
                base_name, query, ['grouped_loader']
            )
            # print(marglik.mean(axis=1).head())
            res = marglik.groupby('grouped_loader').mean()
            t, f = res.loc[True].mean(), res.loc[False].mean()
            print(f'Marglik: {approx} with grouping {t}, without {f} improvement {(f/t-1)*100:.2f}%')
            res = nll.groupby('grouped_loader').mean()
            t, f = res.loc[True].mean(), res.loc[False].mean()
            print(f'NLL: {approx} with grouping {t}, without {f} improvement {(f/t-1)*100:.2f}%')
