"""Numerical calculations used to prepare the saved notebook tables.

Run runs/calculate_results.py explicitly after changing inputs. These functions
return numbers, arrays and tables; figures are drawn only by the notebooks.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import result_tables as th
import final_visualization as fv
from metrics import compute_metrics, existence_auprc, directed_auprc
from statistical_methods import holm, paired_scope
from Wilcoxon_test.wilcoxon_test import paired_wilcoxon as wilcoxon
from varsortability import calculate_varsortability
ROOT = Path(__file__).resolve().parents[1]

def calculate_priors():
    """Calculate prior-quality scores, summaries and the recorded uncertainty interval."""
    from metrics import existence_auprc, directed_auprc
    config = json.loads((ROOT / 'configs/prism_study.json').read_text())
    rows = []
    with np.load(ROOT / 'results/prism/priors/search_estimates.npz', allow_pickle=False) as estimates:
        for family in config['families']:
            for seed in config['search_dataset_seeds']:
                with np.load(ROOT / 'datasets/synthetic' / f'{family}__seed_{int(seed)}.npz', allow_pickle=False) as observations:
                    truth = observations['true_adj']
                prefix = f'{family}__seed_{seed}__'
                values = {name: estimates[prefix + name] for name in ('r_G', 'r_H_raw', 'r_H_z', 'd_binary', 'd_binary_z', 'd_bootstrap', 'd_bootstrap_z')}

                def record(score, representation, matrix, target, alpha=None):
                    metric = existence_auprc if target == 'skeleton' else directed_auprc
                    rows.append(dict(dataset=family, dataset_seed=seed, score=score, representation=representation, alpha=alpha, target=target, auprc=metric(truth, matrix)))
                record('GLasso', 'correlation', values['r_G'], 'skeleton')
                for representation, key in (('raw', 'r_H_raw'), ('standardized', 'r_H_z')):
                    record('Hessian', representation, values[key], 'skeleton')
                    for alpha in (0.25, 0.5, 0.75):
                        combined = alpha * values['r_G'] + (1 - alpha) * values[key]
                        record('Combined', representation, combined, 'skeleton', alpha)
                for label, raw, standardized in (('Binary', 'd_binary', 'd_binary_z'), ('Subsampling', 'd_bootstrap', 'd_bootstrap_z')):
                    record(label, 'raw', values['r_G'] * values[raw], 'directed')
                    record(label, 'standardized', values['r_G'] * values[standardized], 'directed')
    scores = pd.DataFrame(rows)
    prior_table = th.prior_quality(scores)
    mixtures = th.prior_quality(scores, include_extra=True)
    direction = scores.loc[scores.score.eq('Subsampling')].pivot(index=['dataset', 'dataset_seed'], columns='representation', values='auprc')
    direction_summary = direction.groupby('dataset')[['raw', 'standardized']].mean()
    direction_summary.loc['Macro'] = direction.mean()
    differences = [(direction.loc[family, 'raw'] - direction.loc[family, 'standardized']).to_numpy() for family in config['families']]
    rng = np.random.default_rng(20260919)
    draws = np.array([np.mean([rng.choice(d, len(d), replace=True).mean() for d in differences]) for _ in range(5000)])
    lower, upper = np.quantile(draws, [0.025, 0.975])
    confidence_intervals = pd.DataFrame([dict(raw_minus_standardized=float(np.mean(differences)), ci95_lower=lower, ci95_upper=upper)])
    return {'scores': scores, 'prior_table': prior_table, 'mixtures': mixtures, 'direction_summary': direction_summary, 'confidence_intervals': confidence_intervals}


def calculate_tuning():
    """Evaluate saved PRISM predictions and summarize selection and held-out performance."""
    fits = []
    recorded = pd.concat([pd.read_csv(path, low_memory=False, float_precision='round_trip') for path in sorted((ROOT / 'results/prism/tuning').glob('*.csv'))], ignore_index=True)
    for stage, name, threshold in [('search', 'search_trial_0068.npz', 0.25), ('validation', 'validation_candidate_0068.npz', 0.1)]:
        with np.load(ROOT / 'results/prism/tuning/predictions' / name, allow_pickle=False) as matrices:
            for key in matrices.files:
                family, identity = key.split('__ds')
                ds, ms = map(int, identity.split('__ms'))
                with np.load(ROOT / 'datasets/synthetic' / f'{family}__seed_{int(ds)}.npz', allow_pickle=False) as data:
                    truth = data['true_adj']
                    labels = [f'X{i}' for i in range(data['X'].shape[1])]
                graph = (np.abs(matrices[key]) > threshold).astype(int)
                np.fill_diagonal(graph, 0)
                row = compute_metrics(truth, graph, labels)
                p, r = (row['precision'], row['recall'])
                row['fbeta'] = 1.25 * p * r / (0.25 * p + r) if 0.25 * p + r else 0.0
                identity_row = recorded.loc[recorded.dataset.eq(family) & recorded.condition_id.eq(stage) & recorded.dataset_seed.eq(ds) & recorded.model_seed.eq(ms)].iloc[0].to_dict()
                fits.append(dict(**{key: value for key, value in identity_row.items() if key not in row and key not in ('dataset', 'dataset_seed', 'model_seed')}, stage=stage, dataset=family, dataset_seed=ds, model_seed=ms, threshold=threshold, **row))
    sources = th.load_final_results()
    complete = sources['complete']
    final = complete.loc[complete.model.eq('PRISM') & complete.dataset.isin(th.TEST_FAMILIES)].copy()
    final['stage'] = 'test'
    final['threshold'] = 0.1
    stages = pd.concat([pd.DataFrame(fits), final], ignore_index=True)
    blocks = stages.groupby(['stage', 'dataset', 'dataset_seed']).fbeta.mean().reset_index()
    macro = blocks.groupby(['stage', 'dataset_seed']).fbeta.mean().reset_index()
    stage_summary = macro.groupby('stage').fbeta.agg(['mean', 'std', 'size']).reset_index()
    scores = th.seed_means(complete.loc[complete.model.eq('PRISM')], 'fbeta', arms=('tuned',))
    orders = []
    for row in scores.itertuples(index=False):
        family, seed = (row.dataset, int(row.dataset_seed))
        with np.load(ROOT / 'datasets/synthetic' / f'{family}__seed_{int(seed)}.npz', allow_pickle=False) as d:
            truth = d['true_adj']
        with np.load(th.PRIORS / f'{family}__seed_{seed}.npz', allow_pickle=False) as d:
            order = d['order']
        fnr = th.order_fnr(truth, order)
        orders.append(dict(dataset=family, dataset_seed=seed, fbeta=row.fbeta, order_fnr=fnr))
    orders = pd.DataFrame(orders)
    associations = []
    for family, group in orders.groupby('dataset'):
        result = spearmanr(group.order_fnr, group.fbeta)
        associations.append(dict(dataset=family, n_pairs=len(group), spearman_rho=float(result.statistic), pvalue=float(result.pvalue)))
    correlations = pd.DataFrame(associations)
    prism_table = th.prism_performance(complete)
    return {'stage_summary': stage_summary, 'correlations': correlations, 'prism_table': prism_table}


def calculate_statistics():
    """Calculate paired method comparisons, adjusted p-values and ablation comparisons."""
    sources = th.load_final_results()
    main = sources['benchmark_with_biological']
    complete = sources['complete']
    collected = [[] for _ in range(5)]

    def add(frame, experiment, scope, metric, higher=True):
        for target, table in zip(collected, paired_scope(frame, experiment, scope, metric, higher)):
            table['experiment'] = experiment
            table['scope'] = scope
            table['metric_name'] = metric
            target.append(table)
    for metric, higher in [('f1', True), ('SHD', False), ('fbeta', True)]:
        for family, frame in main.groupby('dataset', observed=True):
            add(frame, 'benchmark', family, metric, higher)
        for arm in ('tuned', 'untuned'):
            selected = complete.loc[complete.arm.eq(arm)]
            for family, frame in selected.groupby('dataset', observed=True):
                add(frame, arm, family, metric, higher)
            for scope, families in [('Test', th.TEST_FAMILIES), ('OOD', th.OOD_FAMILIES), ('Test + OOD', th.SIX_FAMILIES)]:
                add(selected.loc[selected.dataset.isin(families)], arm, scope, metric, higher)
    names = ['paired_results', 'friedman', 'rankings', 'coverage', 'dataset_blocks']
    comparison_tables = {name: pd.concat(pieces, ignore_index=True) for name, pieces in zip(names, collected)}
    gains = []
    for metric in ('f1', 'fbeta', 'SHD'):
        for family, group in complete.groupby('dataset', observed=True):
            for model, frame in group.groupby('model', observed=True):
                if not frame.finished.all():
                    continue
                cells = frame.groupby(['dataset_seed', 'arm'])[metric].mean().unstack('arm')
                if not {'tuned', 'untuned'} <= set(cells):
                    continue
                cells = cells.dropna(subset=['tuned', 'untuned'])
                delta = cells.tuned - cells.untuned
                r = None if (delta == 0).all() else wilcoxon(delta)
                gains.append(dict(dataset=family, model=model, metric=metric, n_pairs=len(delta), mean_tuned=float(cells.tuned.mean()), mean_untuned=float(cells.untuned.mean()), mean_difference=float(delta.mean()), statistic=0.0 if r is None else float(r.statistic), pvalue=1.0 if r is None else float(r.pvalue)))
    gains = pd.DataFrame(gains)
    gains['pvalue_holm'] = gains.groupby(['dataset', 'metric']).pvalue.transform(lambda v: holm(v))
    gains['significant_holm'] = gains.pvalue_holm < 0.05
    favorable = np.where(gains.metric.eq('SHD'), gains.mean_difference < 0, gains.mean_difference > 0)
    gains['winner'] = np.where(gains.significant_holm, np.where(favorable, 'tuned', 'untuned'), 'no_significant_difference')
    gain_family = th.TEST_FAMILIES[0]
    fits = pd.concat([pd.read_csv(path, low_memory=False, float_precision='round_trip') for path in sorted((ROOT / 'results/prism/ablation').glob('*.csv'))], ignore_index=True).assign(family=lambda frame: frame.dataset)
    fits = fits.loc[fits.status.eq('finished') & fits.fbeta.notna()]
    cells = fits.groupby(['condition_id', 'family', 'dataset_seed']).fbeta.mean().reset_index()
    macro = cells.groupby(['condition_id', 'dataset_seed']).filter(lambda g: g.family.nunique() == 3).groupby(['dataset_seed', 'condition_id']).fbeta.mean().unstack('condition_id')
    comparisons = []
    for condition in macro.columns:
        if condition == 'full':
            continue
        pair = macro[['full', condition]].dropna()
        delta = pair['full'] - pair[condition]
        r = None if (delta == 0).all() else wilcoxon(delta)
        comparisons.append(dict(condition=condition, n_pairs=len(pair), mean_full=float(pair['full'].mean()), mean_condition=float(pair[condition].mean()), loss=float(delta.mean()), statistic=0.0 if r is None else float(r.statistic), pvalue=1.0 if r is None else float(r.pvalue)))
    table = pd.DataFrame(comparisons)
    table['pvalue_holm'] = holm(table.pvalue)
    table['winner'] = np.where(table.pvalue_holm < 0.05, np.where(table.loss > 0, 'full', table.condition), 'no_significant_difference')
    comparison_tables.update(tuning_comparisons=gains, ablation_comparisons=table)
    summary = comparison_tables['paired_results'].groupby(['experiment', 'metric']).agg(contexts=('scope', 'nunique'), comparisons=('n_pairs', 'size'), significant_comparisons=('significant_holm', 'sum')).reset_index()
    return {'comparison_tables': comparison_tables, 'summary': summary}


def calculate_benchmark():
    """Prepare performance, coverage, rankings and seed-variability tables."""
    sources = th.load_final_results()
    benchmark = sources['benchmark']
    benchmark_biological = sources['benchmark_biological']
    benchmark_with_biological = sources['benchmark_with_biological']
    benchmark_rank_table = th.benchmark_ranks(benchmark, benchmark_biological)
    benchmark_cells = th.seed_means(benchmark_with_biological, 'f1')
    benchmark_f1 = benchmark_cells.groupby(['dataset', 'model'], observed=True)['f1'].agg(['mean', 'std', 'size']).reset_index()
    benchmark_coverage = benchmark_with_biological.groupby(['dataset', 'model'], observed=True).agg(planned_fits=('finished', 'size'), completed_fits=('finished', 'sum')).reset_index()
    complete = sources['complete']
    tuned_table = th.tuned_test_ood(complete)
    tuned_ranks = th.ranks_by_family(complete, metric='fbeta')
    tuned_scope_ranks = th.ranks_by_scope(complete, metric='fbeta')
    main = benchmark_with_biological
    runtime_tables, coverage_tables = ({}, {})
    runtime_tables = {}
    for name, frame in [('benchmark', main), ('tuned_untuned', complete)]:
        keys = ['dataset', 'model'] + (['arm'] if name == 'tuned_untuned' else [])
        coverage = frame.groupby(keys).agg(recorded=('finished', 'size'), completed=('finished', 'sum')).reset_index()
        coverage['completion_fraction'] = coverage.completed / coverage.recorded
        coverage_tables[name] = coverage
        timings = frame.loc[frame.finished].groupby(keys + ['dataset_seed']).runtime_seconds.mean().reset_index()
        runtime_tables[name] = timings.groupby(keys).runtime_seconds.agg(['mean', 'std', 'size']).reset_index()
    frame = th.prepare_metrics(pd.concat([th.load_graph_size_results(), benchmark_with_biological], ignore_index=True))
    metrics = ['f1', 'fbeta', 'precision', 'recall', 'SHD', 'runtime_seconds']
    blocks = frame.loc[frame.finished].groupby(['dataset', 'n_nodes', 'dataset_seed', 'model'], observed=True)[metrics].mean().reset_index()
    summary = blocks.groupby(['dataset', 'n_nodes', 'model'], observed=True)[metrics].agg(['mean', 'std', 'count'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()
    coverage = frame.groupby(['dataset', 'n_nodes', 'model'], observed=True).agg(recorded=('finished', 'size'), completed=('finished', 'sum')).reset_index()
    coverage['completion_fraction'] = coverage.completed / coverage.recorded
    metrics_summary = summary
    finished = frame.loc[frame.finished]
    model_seed = finished.groupby(['dataset', 'n_nodes', 'dataset_seed', 'model'], observed=True).f1.agg(lambda v: v.max() - v.min()).rename('f1_range').reset_index()
    dataset_seed = finished.groupby(['dataset', 'n_nodes', 'model_seed', 'model'], observed=True).f1.agg(lambda v: v.max() - v.min()).rename('f1_range').reset_index()
    model_summary = model_seed.groupby(['dataset', 'n_nodes', 'model'], observed=True).f1_range.agg(['mean', 'std', 'size']).reset_index()
    dataset_summary = dataset_seed.groupby(['dataset', 'n_nodes', 'model'], observed=True).f1_range.agg(['mean', 'std', 'size']).reset_index()
    metric_summaries = {}
    for metric in ('f1', 'SHD', 'precision', 'recall'):
        blocks = frame.loc[frame.finished].groupby(['dataset', 'n_nodes', 'dataset_seed', 'model'], observed=True)[metric].mean().reset_index()
        metric_summaries[metric] = blocks.groupby(['dataset', 'n_nodes', 'model'], observed=True)[metric].agg(['mean', 'std', 'size']).reset_index()
    size_tables = {}
    for size in sorted(frame.n_nodes.dropna().unique()):
        selected = frame.loc[frame.n_nodes.eq(size) & frame.finished & ~frame.dataset.str.contains('HighGaussian_Linear')]
        for metric in ('f1', 'SHD'):
            blocks = selected.groupby(['dataset', 'dataset_seed', 'model'])[metric].mean().reset_index()
            tab = blocks.groupby(['dataset', 'model'])[metric].agg(['mean', 'std', 'size']).reset_index()
            tab['score'] = tab.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
            size_tables[size, metric] = tab.pivot(index='model', columns='dataset', values='score')
    model_order = benchmark_with_biological.loc[benchmark_with_biological.finished & benchmark_with_biological.dataset.isin(th.SIX_FAMILIES)].groupby(['dataset', 'dataset_seed', 'model'], observed=True)['f1'].mean().groupby(['dataset', 'model'], observed=True).mean().groupby('model', observed=True).mean().sort_values(ascending=False)
    return {'benchmark_rank_table': benchmark_rank_table, 'benchmark_f1': benchmark_f1, 'benchmark_coverage': benchmark_coverage, 'tuned_table': tuned_table, 'tuned_ranks': tuned_ranks, 'tuned_scope_ranks': tuned_scope_ranks, 'runtime_tables': runtime_tables, 'coverage_tables': coverage_tables, 'metrics_summary': metrics_summary, 'coverage': coverage, 'model_summary': model_summary, 'dataset_summary': dataset_summary, 'model_seed': model_seed, 'dataset_seed': dataset_seed, 'metric_summaries': metric_summaries, 'size_tables': size_tables, 'model_order': model_order}


def calculate_control():
    """Summarize the Gaussian-linear control and comparative benchmark measurements."""
    control = pd.concat([pd.read_csv(path, low_memory=False, float_precision='round_trip') for path in sorted((ROOT / 'results/benchmark/control').glob('*.csv'))], ignore_index=True)
    reference = th.load_final_results()['benchmark_with_biological']
    reference = reference.loc[reference.dataset.eq('20nodes_ER_obsconf0_latconf0_Gaussian_Linear')]
    control['setting'] = 'High variance Gaussian'
    reference['setting'] = 'Gaussian'
    runs = pd.concat([reference, control], ignore_index=True)
    blocks = runs.loc[runs.status.eq('finished')].groupby(['setting', 'model', 'dataset_seed']).f1.mean().reset_index()
    summary = blocks.groupby(['setting', 'model']).f1.agg(['mean', 'std', 'size']).reset_index()
    wide = summary.pivot(index='model', columns='setting', values='mean')
    return {'summary': summary, 'wide': wide}


def calculate_sortability():
    """Calculate sortability, associations and available sorting-baseline scores."""
    def sortability(frame, study):
        rows = []
        for family, seed in frame[['dataset', 'dataset_seed']].drop_duplicates().itertuples(index=False, name=None):
            path = ROOT / 'datasets/synthetic' / f'{family}__seed_{int(seed)}.npz'
            with np.load(path, allow_pickle=False) as data:
                raw = 'X_raw' in data.files
                result = calculate_varsortability(data['X_raw'] if raw else data['X'], data['true_adj'])
                if not raw:
                    for key in ('varsortability', 'n_ordered_variance_paths', 'n_tied_variance_paths'):
                        result[key] = np.nan
            rows.append(dict(dataset=family, dataset_seed=seed, study=study, raw_variance_available=raw, sample_path=str(path.relative_to(ROOT)), **result))
        return pd.DataFrame(rows)
    sources = th.load_final_results()
    frames = [('benchmark', th.prepare_metrics(pd.concat([th.load_graph_size_results(), th.load_final_results()['benchmark_with_biological']], ignore_index=True))), ('tuned_untuned', sources['complete'])]
    sort_tables, correlations = ([], [])
    for study, frame in frames:
        values = sortability(frame, study)
        sort_tables.append(values)
        arms = [('untuned', frame)] if study == 'benchmark' else frame.groupby('arm', observed=True)
        for arm, arm_frame in arms:
            blocks = th.seed_means(arm_frame, 'f1').merge(values, on=['dataset', 'dataset_seed'])
            for (family, model), group in blocks.groupby(['dataset', 'model'], observed=True):
                for score in ('r2_sortability', 'varsortability'):
                    valid = group[['f1', score]].dropna()
                    if len(valid) < 3:
                        continue
                    result = None if valid.nunique().min() < 2 else spearmanr(valid[score], valid.f1)
                    correlations.append(dict(study=study, arm=arm, dataset=family, model=model, sortability=score, n_pairs=len(valid), rho=np.nan if result is None else result.statistic, pvalue=np.nan if result is None else result.pvalue, status='constant_or_insufficient' if result is None else 'estimated'))
    sortability_all = pd.concat(sort_tables, ignore_index=True)
    corr = pd.DataFrame(correlations)
    corr['pvalue_holm'] = np.nan
    valid = corr.pvalue.notna()
    corr.loc[valid, 'pvalue_holm'] = corr.loc[valid].groupby(['study', 'arm', 'sortability']).pvalue.transform(holm)
    sortability_all['n_nodes'] = sortability_all.dataset.str.extract('^(\\d+)nodes').astype(int)
    family_summary = sortability_all.groupby(['study', 'dataset', 'n_nodes']).agg(dataset_instances=('dataset_seed', 'nunique'), varsortability_mean=('varsortability', 'mean'), varsortability_std=('varsortability', 'std'), r2_sortability_mean=('r2_sortability', 'mean'), r2_sortability_std=('r2_sortability', 'std')).reset_index()
    gaussian = th.raw_sortability()
    r2_baseline = th.r2_sort_regress()
    from varsortability import calculate_r2_sort_regress_table
    raw_instances = sortability_all.loc[sortability_all.raw_variance_available].drop_duplicates(['dataset', 'dataset_seed'])
    r2_baseline_all = calculate_r2_sort_regress_table(raw_instances, project_root=ROOT)
    r2_baseline_summary = r2_baseline_all.groupby('dataset').f1.agg(['mean', 'std', 'size']).reset_index()
    return {'sortability_all': sortability_all, 'corr': corr, 'family_summary': family_summary, 'gaussian': gaussian, 'r2_baseline': r2_baseline, 'r2_baseline_all': r2_baseline_all, 'r2_baseline_summary': r2_baseline_summary}


def calculate_tcga_biological():
    """Summarize reference recovery, candidate edges and TCGA threshold sensitivity."""
    from glioma_analysis import build_internal_threshold_graphs, build_model_graphs, import_existing_analyses, internal_threshold_sensitivity_summary, load_glioma_results
    tcga_models = ('PRISM', 'GRASP', 'AVICI', 'PC', 'GES', 'GOLEM')
    glioma = load_glioma_results(ROOT, tcga_models)
    tcga_graphs = build_model_graphs(glioma, tcga_models, min_rate=0.8, edge_mode='directed')
    primary_graphs = tcga_graphs['PRISM']
    contexts = ['Full cohort', 'Astrocytoma', 'Glioblastoma', 'Oligodendroglioma']
    context_labels = contexts.copy()
    joint_datasets = ['tcga_mut_expr70', 'tcga_mut_expr70__astro', 'tcga_mut_expr70__gbm', 'tcga_mut_expr70__oligo']
    phased_scorer = import_existing_analyses(glioma)['phased_scoring']
    reference_genes = sorted({label.split('_', 1)[0] for label in glioma.labels['tcga_mut_expr70']})
    relations = phased_scorer.load_relations(reference_genes)
    control_edges = sorted({(row['source'], row['target']) for row in relations})

    def edge_rate(graph, source, target):
        rows = graph.loc[(graph.source == source) & (graph.target == target)]
        return float(rows.rate.max()) if not rows.empty else 0.0
    rates = np.zeros((len(control_edges), len(joint_datasets)))
    support = np.zeros_like(rates, dtype=int)
    for i, (source, target) in enumerate(control_edges):
        for j, dataset in enumerate(joint_datasets):
            rates[i, j] = edge_rate(primary_graphs[dataset], source, target)
            support[i, j] = sum((edge_rate(graphs[dataset], source, target) >= 0.8 for model, graphs in tcga_graphs.items() if model != 'PRISM' and dataset in graphs))

    def clean_variable(value):
        return value.replace('_expr_log2tpm', ' RNA').replace('_mut', ' mut.')
    labels = [f'{clean_variable(source)} → {clean_variable(target)}' for source, target in control_edges]
    thresholds = (0.1, 0.125, 0.15, 0.2)
    matrix_roots = (ROOT / 'results/tcga/arms_results', ROOT / 'results/tcga/arms_results_subtypes')
    threshold_graphs, _ = build_internal_threshold_graphs(glioma, thresholds=thresholds, consensus_rate=0.8, model='PRISM', matrix_roots=matrix_roots)
    _, reference_detail = internal_threshold_sensitivity_summary(glioma, threshold_graphs, phase=1)

    def edge_set(frame):
        return set(zip(frame.source, frame.target))
    rows = []
    for threshold in thresholds:
        expr = edge_set(threshold_graphs[threshold]['tcga_expr35'])
        joint_frame = threshold_graphs[threshold]['tcga_mut_expr70']
        joint_expr = {(s, t) for s, t in edge_set(joint_frame) if not s.endswith('_mut') and (not t.endswith('_mut'))}
        rows.append({'internal_threshold': threshold, 'expression_only': len(expr), 'mutation_expression': len(joint_expr)})
    threshold_edges = pd.DataFrame(rows)
    reference_summary = reference_detail[reference_detail.dataset.isin(('tcga_expr35', 'tcga_mut_expr70'))].groupby(['internal_threshold', 'dataset'], as_index=False).agg(applicable=('status', 'size'), recovered=('status', lambda x: x.eq('direta_sentido_certo').sum()))
    reference_summary['recall'] = 100 * reference_summary.recovered / reference_summary.applicable
    from select_tcga_high_confidence_edges import EXPRESSION_FAMILY, JOINT_FAMILY, family_candidates, reference_edges, short_name
    expression_candidates = family_candidates(tcga_graphs, EXPRESSION_FAMILY, 'expr->expr')
    joint_candidates = family_candidates(tcga_graphs, JOINT_FAMILY, 'expr->expr')
    known = reference_edges()
    candidate_rows = []
    for edge in sorted(set(expression_candidates) & set(joint_candidates)):
        if edge in known or (edge[1], edge[0]) in known:
            continue
        supporting = sorted(set(expression_candidates[edge]['supporting_methods']) | set(joint_candidates[edge]['supporting_methods']))
        candidate_rows.append({'Candidate relationship': f'{short_name(edge[0])} -> {short_name(edge[1])}', 'Expr.': f"{expression_candidates[edge]['n_contexts']}/4", 'Joint': f"{joint_candidates[edge]['n_contexts']}/4", 'Min. seed support': min(expression_candidates[edge]['minimum_seed_rate'], joint_candidates[edge]['minimum_seed_rate']), 'Supporting methods': ', '.join((th.model_label(model) for model in supporting))})
    candidate_table = pd.DataFrame(candidate_rows).sort_values(['Expr.', 'Joint', 'Min. seed support'], ascending=False)
    expression = edge_set(threshold_graphs[0.1]['tcga_expr35'])
    joint = edge_set(threshold_graphs[0.1]['tcga_mut_expr70'])
    joint_expression = {(s, t) for s, t in joint if not s.endswith('_mut') and (not t.endswith('_mut'))}
    overlap = pd.DataFrame([dict(expression_edges=len(expression), joint_total_edges=len(joint), joint_expression_edges=len(joint_expression), shared_direction=len(expression & joint_expression), expression_only=len(expression - joint_expression), joint_only=len(joint_expression - expression), net_expression_edges=len(joint_expression) - len(expression))])
    selected = (set(expression_candidates) & set(joint_candidates)) - known
    selected = {edge for edge in selected if (edge[1], edge[0]) not in known}
    strict_graphs, _ = build_internal_threshold_graphs(glioma, thresholds=(0.1,), consensus_rate=1.0, model='PRISM', matrix_roots=matrix_roots)
    _, strict_detail = internal_threshold_sensitivity_summary(glioma, strict_graphs, phase=1)
    rows = []
    for agreement, graphs, detail in ((0.8, threshold_graphs[0.1], reference_detail.loc[reference_detail.internal_threshold.eq(0.1)]), (1.0, strict_graphs[0.1], strict_detail)):
        for dataset in ('tcga_expr35', 'tcga_mut_expr70'):
            edges = edge_set(graphs[dataset])
            expression_edges = {(s, t) for s, t in edges if not s.endswith('_mut') and (not t.endswith('_mut'))}
            references = detail.loc[detail.dataset.eq(dataset)]
            rows.append(dict(seed_agreement=agreement, dataset=dataset, total_edges=len(edges), expression_edges=len(expression_edges), candidates_retained=len(selected & edges), references_recovered=int(references.status.eq('direta_sentido_certo').sum()), references_applicable=len(references)))
    consensus = pd.DataFrame(rows)
    retention = []
    for threshold, contexts in threshold_graphs.items():
        for dataset, graph in contexts.items():
            if dataset.startswith('tcga_mut35'):
                continue
            edges = edge_set(graph)
            for source, target in sorted(selected):
                retention.append(dict(relationship=f'{short_name(source)} -> {short_name(target)}', dataset=dataset, internal_threshold=threshold, present=(source, target) in edges))
    retention = pd.DataFrame(retention)
    retention_summary = retention.groupby(['relationship', 'internal_threshold']).present.sum().unstack('internal_threshold')
    return {'rates': rates, 'support': support, 'contexts': context_labels, 'labels': labels, 'thresholds': thresholds, 'threshold_edges': threshold_edges, 'reference_summary': reference_summary, 'candidate_table': candidate_table, 'overlap': overlap, 'consensus': consensus, 'retention_summary': retention_summary}


def calculate_tcga_agreement():
    """Summarize method, cohort and subtype agreement for the TCGA datasets."""
    import glioma_analysis as ga
    models = ('PRISM', 'GRASP', 'AVICI', 'PC', 'GES', 'GOLEM')
    bundle = ga.load_glioma_results(ROOT, models)
    graphs = ga.build_model_graphs(bundle, models, min_rate=0.8, edge_mode='directed')
    models = tuple(graphs)
    agreements = []
    cohort = []
    counts = []
    for dataset in bundle.datasets:
        sets = {m: set(zip(graphs[m][dataset].source, graphs[m][dataset].target)) for m in models}
        for a, b in combinations(models, 2):
            union = sets[a] | sets[b]
            agreements.append(dict(dataset=dataset, model_a=a, model_b=b, common_edges=len(sets[a] & sets[b]), union_edges=len(union), jaccard=len(sets[a] & sets[b]) / len(union) if union else np.nan))
        for m, edges in sets.items():
            for kind in ('mut->mut', 'mut->rna', 'rna->rna', 'rna->mut'):
                a, b = kind.split('->')
                count = sum((ga.variable_kind(s) == a and ga.variable_kind(t) == b for s, t in edges))
                counts.append(dict(dataset=dataset, model=m, edge_kind=kind, n_edges=count))
            if '__' in dataset:
                base = dataset.split('__')[0]
                ref = set(zip(graphs[m][base].source, graphs[m][base].target))
                union = ref | edges
                cohort.append(dict(dataset=dataset, model=m, cohort_edges=len(ref), subtype_edges=len(edges), common_edges=len(ref & edges), jaccard=len(ref & edges) / len(union) if union else np.nan))
    reproducibility = pd.DataFrame(cohort)
    threshold_graphs, _ = ga.build_internal_threshold_graphs(bundle, thresholds=(0.1, 0.125, 0.15, 0.2), consensus_rate=0.8, model='PRISM', matrix_roots=(ROOT / 'results/tcga/arms_results', ROOT / 'results/tcga/arms_results_subtypes'))
    summary, detail = ga.internal_threshold_sensitivity_summary(bundle, threshold_graphs, phase=1)
    threshold_counts = pd.DataFrame([dict(threshold=t, dataset=d, n_edges=len(g)) for t, context in threshold_graphs.items() for d, g in context.items()])
    return {'reproducibility': reproducibility, 'summary': summary, 'detail': detail, 'threshold_counts': threshold_counts}


def calculate_sachs():
    """Prepare Sachs performance summaries by method and configuration."""
    import sachs_table
    sachs_runs = sachs_table.load()
    sachs_summary = sachs_runs.groupby(['base', 'arm'], observed=True)['fbeta'].agg(mean='mean', lower=lambda x: x.quantile(0.25), upper=lambda x: x.quantile(0.75)).reset_index()
    models = sachs_summary.groupby('base', observed=True)['mean'].mean().sort_values(ascending=False).index.tolist()
    sachs_long, sachs_wide = sachs_table.build()
    sachs_clean = sachs_long.drop(columns=[c for c in sachs_long if c.startswith('_')])
    sachs_clean['Modelo'] = sachs_clean['Modelo'].map(th.model_label)
    sachs_untuned = sachs_clean[sachs_clean['Braço'].eq('untuned')].copy()
    metric_summaries = {metric: sachs_runs.groupby(['base', 'arm'])[metric].agg(['mean', 'std', 'size']).reset_index() for metric in ('f1', 'fbeta', 'precision', 'recall', 'SHD')}
    return {'sachs_summary': sachs_summary, 'models': models, 'sachs_clean': sachs_clean, 'metric_summaries': metric_summaries}


def calculate_scalability():
    """Prepare accuracy, runtime and completion tables across reported graph sizes."""
    scalability = th.load_scalability_results()
    scalability_finished = scalability.loc[scalability.status.eq('finished')].copy()
    scale_cells = scalability_finished.groupby(['scalability_family', 'n_nodes', 'dataset_seed', 'model'], observed=True)['f1'].mean().reset_index()
    scale_summary = scale_cells.groupby(['scalability_family', 'n_nodes', 'model'], observed=True)['f1'].agg(['mean', 'std', 'size']).reset_index()
    scale_summary['F1'] = scale_summary.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)
    scale_gaussian = scale_summary.loc[scale_summary.scalability_family.eq('er_gaussian_linear')]
    scale_wide = scale_gaussian.pivot(index='model', columns='n_nodes', values='mean')
    scale_wide = scale_wide.reindex(index=['PC', 'FCI', 'GES', 'GRaSP', 'DirectLiNGAM', 'NOTEARS', 'GOLEM', 'DAGMA_Linear', 'SCORE', 'NoGAM', 'DAS', 'GraNDAG', 'DAGMA_Nonlinear', 'AVICI', 'Random'], columns=[10, 15, 20, 30, 70, 100, 150, 250, 500])
    scale_wide = scale_wide.map(lambda value: '--' if pd.isna(value) else f'{value:.2f}')
    scale_wide.index = [th.model_label(model) for model in scale_wide.index]
    scale_wide = scale_wide.reset_index(names='Model')
    mixed = scale_summary.loc[scale_summary.scalability_family.eq('er_nongaussian_mixed')]
    mixed_wide = mixed.pivot(index='model', columns='n_nodes', values='mean')
    mixed_wide = mixed_wide.reindex(index=[m for m in fv.DEFAULT_MODEL_ORDER if m != 'Random'], columns=[10, 15, 20, 30, 70, 100, 150, 250, 500])
    mixed_wide = mixed_wide.map(lambda v: '—' if pd.isna(v) else f'{v:.3f}')
    mixed_wide.index = [th.model_label(m) for m in mixed_wide.index]
    mixed_wide = mixed_wide.reset_index(names='Method')
    progression = scalability.loc[scalability.n_nodes.ge(70)].groupby(['n_nodes', 'model'], observed=True)['status'].agg(recorded='size', completed=lambda s: s.eq('finished').sum()).reset_index()
    progression['planned'] = 30
    progression['advance_to_next_size'] = progression.completed.ge(10).astype('boolean')
    progression['next_size'] = progression.n_nodes.map({70: 100, 100: 150, 150: 250, 250: 500})
    progression.loc[progression.next_size.isna(), 'advance_to_next_size'] = pd.NA
    metric_summaries = {}
    for metric in ('f1', 'runtime_seconds'):
        cells = scalability.loc[scalability.status.eq('finished')].groupby(['scalability_family', 'n_nodes', 'dataset_seed', 'model'])[metric].mean().reset_index()
        metric_summaries[metric] = cells.groupby(['scalability_family', 'n_nodes', 'model'])[metric].agg(['mean', 'std', 'size']).reset_index()
    return {'scale_summary': scale_summary, 'scale_wide': scale_wide, 'mixed_wide': mixed_wide, 'progression': progression, 'metric_summaries': metric_summaries}


def calculate_parameters():
    """Format the recorded search spaces and selected model parameters."""
    baseline = json.loads((ROOT / 'configs/baseline_study.json').read_text())
    rows = []
    for model, space in baseline['spaces'].items():
        if not space:
            fixed = baseline['defaults'][model] or {'pretrained': 'scm-v0'}
            for parameter, value in fixed.items():
                rows.append(dict(model=model, trials=baseline['trials'][model], parameter=parameter, default=value, search=json.dumps([value]), sampling='fixed'))
        for parameter, definition in space.items():
            options = definition.get('choices', [definition.get('low'), definition.get('high')])
            rows.append(dict(model=model, trials=baseline['trials'][model], parameter=parameter, default=baseline['defaults'][model].get(parameter), search=json.dumps(options), sampling=definition['type']))
    spaces = pd.DataFrame(rows)
    winners = json.loads((ROOT / 'configs/frozen/frozen_winners.json').read_text())['winners']
    prism = json.loads((ROOT / 'configs/prism_winner.json').read_text())
    rows = [dict(model=model, threshold=winner.get('threshold'), parameters=json.dumps(winner['params'], sort_keys=True)) for model, winner in winners.items()]
    rows.append(dict(model='PRISM', threshold=prism['threshold'], parameters=json.dumps(prism['params'], sort_keys=True)))
    selected = pd.DataFrame(rows)
    return {'spaces': spaces, 'selected': selected}


def calculate_ablation():
    """Summarize the recorded PRISM component ablations."""
    ablation_table = th.ablation_table()
    return {'ablation_table': ablation_table}


def calculate_families():
    """Prepare family-level metrics and variability summaries by configuration."""
    sources = th.load_final_results()
    tables = {}
    for study, frame in [('benchmark', sources['benchmark_with_biological']), ('tuned', sources['complete'])]:
        for (family, arm), group in frame.groupby(['dataset', 'arm'], observed=True):
            ok = group.loc[group.finished]
            summaries = {}
            for metric in ('f1', 'SHD', 'precision', 'recall'):
                blocks = ok.groupby(['dataset_seed', 'model'])[metric].mean().reset_index()
                summaries[metric] = blocks.groupby('model')[metric].agg(['mean', 'std', 'size']).reset_index()
            ranges = {
                'model_seed': ok.groupby(['dataset_seed', 'model']).f1.agg(lambda v: v.max() - v.min()).rename('f1_range').reset_index(),
                'dataset_seed': ok.groupby(['model_seed', 'model']).f1.agg(lambda v: v.max() - v.min()).rename('f1_range').reset_index(),
            }
            range_summaries = {key: value.groupby('model').f1_range.mean().sort_values() for key, value in ranges.items()}
            tables[(study, family, arm)] = dict(summaries=summaries, ranges=ranges, range_summaries=range_summaries)
    return tables


def calculate_convergence():
    """Prepare numerical convergence trajectories, summaries and diagnostics."""
    from prism_convergence_figure import load_convergence, _prepare_traces, _trajectory_summary, convergence_statistics
    fits, traces = load_convergence()
    prepared = _prepare_traces(traces)
    objective = _trajectory_summary(prepared, 'relative_objective')
    acyclicity = _trajectory_summary(prepared, 'h')
    stats = convergence_statistics(fits, prepared)
    points = []
    for path in sorted((ROOT / 'results/prism/convergence/fits').rglob('*.json')):
        payload = json.loads(path.read_text())
        for point in (payload.get('convergence') or {}).get('trace', []):
            points.append(dict(fit=str(path.relative_to(ROOT)), family=payload['family'], **point))
    raw = pd.DataFrame(points)
    raw['start'] = raw.groupby(['fit', 'outer_step', 'attempt']).objective.transform('first')
    raw['relative_objective'] = raw.objective / raw.start
    raw['progress'] = raw.outer_step + raw.iteration / raw.outer_step.map(lambda step: 5000 if step == 5 else 3125)
    family_summary = raw.groupby(['family', 'progress'])[['relative_objective', 'h']].quantile([0.25, 0.5, 0.75]).unstack()
    family_summary.columns = ['_'.join(map(str, col)) for col in family_summary.columns]
    return dict(fits=fits, traces=prepared, objective=objective, acyclicity=acyclicity, stats=stats, family_summary=family_summary.reset_index())


def calculate_plot_data():
    """Prepare regression summaries and assumption contrasts for notebook figures."""
    sources = th.load_final_results()
    sort = pd.read_pickle(ROOT / 'results/analysis/sortability.pkl')
    scatter = {}
    for study, frame in [('benchmark', sources['benchmark_with_biological']), ('tuned', sources['complete'])]:
        tag = 'benchmark' if study == 'benchmark' else 'tuned_untuned'
        for (family, arm), group in frame.groupby(['dataset', 'arm'], observed=True):
            values = sort['sortability_all']
            values = values.loc[values.dataset.eq(family) & values.study.eq(tag)]
            for score in ('r2_sortability', 'varsortability'):
                if values[score].notna().sum() < 3:
                    continue
                scatter[(study, family, arm, score)] = fv.prepare_varsortability_data(
                    th.visualization_data(group), metric='f1', datasets=family,
                    varsortability_values=values, sortability=score, coverage_policy='all',
                    include_r2_sort_regress=False, obs_conf=None, lat_conf=None,
                )
    family = th.OOD_FAMILIES[0]
    scatter['main'] = fv.prepare_varsortability_data(
        sources['benchmark_data'], metric='f1', sortability='r2_sortability',
        include_r2_sort_regress=True, varsortability_values=sort['gaussian'],
        r2_sort_regress_values=sort['r2_baseline'], datasets=family, coverage_policy='complete',
    )
    assumptions = fv.build_assumption_degradation_data(
        sources['benchmark_data'], metric='f1', assumptions=fv.DEFAULT_ASSUMPTIONS, uncertainty='sd',
    )
    return dict(scatter=scatter, assumptions=assumptions)
