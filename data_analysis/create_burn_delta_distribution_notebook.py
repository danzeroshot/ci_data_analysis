
#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import json
import math
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import stats

CSV_PATH = Path('ci_payment_details_2.csv')
OUT_PATH = Path('burn_delta_distribution_analysis.ipynb')


def md_cell(source):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source}


def code_cell(source):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}


def fmt(x, places=4):
    if x is None or not np.isfinite(x):
        return ''
    return f'{float(x):,.{places}f}'


def fmt_int(x):
    return f'{int(x):,}'


def pct(x):
    return f'{100 * float(x):.4f}%'


def esc(x):
    return str(x).replace('|', '\\|').replace('\n', '<br>')


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(esc(v) for v in row) + ' |')
    return '\n'.join(out)


def svg_data_uri(svg):
    data = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f'data:image/svg+xml;base64,{data}'


def load_data():
    ratio = []
    delta_pct = []
    delta_abs = []
    base = []
    burn = []
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            b = float(Decimal(row['CI_BURN_RATE_MONTHS']))
            y = float(Decimal(row['THIS_BURN']))
            r = y / b
            ratio.append(r)
            delta_pct.append(100 * (r - 1))
            delta_abs.append(float(Decimal(row['BURN_DELTA_MONTHS'])))
            base.append(b)
            burn.append(y)
    return np.array(ratio), np.array(delta_pct), np.array(delta_abs), np.array(base), np.array(burn)


def fit_candidates(pos):
    candidates = []
    specs = [
        ('Burr XII', stats.burr12, {'floc': 0}, 3),
        ('Log-logistic / Fisk', stats.fisk, {'floc': 0}, 2),
        ('Lognormal', stats.lognorm, {'floc': 0}, 2),
        ('Weibull', stats.weibull_min, {'floc': 0}, 2),
        ('Gamma', stats.gamma, {'floc': 0}, 2),
    ]
    probs = np.array([.01, .05, .10, .25, .50, .75, .90, .95, .99])
    emp = np.quantile(pos, probs)
    for name, dist, kwargs, k in specs:
        params = dist.fit(pos, **kwargs)
        ll = float(np.sum(dist.logpdf(pos, *params)))
        aic = 2 * k - 2 * ll
        ks = stats.kstest(pos, dist.cdf, args=params).statistic
        pred = dist.ppf(probs, *params)
        log_qrmse = float(np.sqrt(np.mean((np.log(pred) - np.log(emp)) ** 2)))
        candidates.append({'name': name, 'dist': dist, 'params': params, 'k': k, 'll': ll, 'aic': aic, 'ks': ks, 'log_qrmse': log_qrmse, 'emp_q': emp, 'fit_q': pred})
    return sorted(candidates, key=lambda c: c['aic'])


def fit_negative(negmag):
    candidates = []
    specs = [
        ('Weibull', stats.weibull_min, {'floc': 0}, 2),
        ('Gamma', stats.gamma, {'floc': 0}, 2),
        ('Burr XII', stats.burr12, {'floc': 0}, 3),
        ('Log-logistic / Fisk', stats.fisk, {'floc': 0}, 2),
        ('Lognormal', stats.lognorm, {'floc': 0}, 2),
    ]
    probs = np.array([.05, .10, .25, .50, .75, .90, .95])
    emp = np.quantile(negmag, probs)
    for name, dist, kwargs, k in specs:
        params = dist.fit(negmag, **kwargs)
        ll = float(np.sum(dist.logpdf(negmag, *params)))
        aic = 2 * k - 2 * ll
        ks = stats.kstest(negmag, dist.cdf, args=params).statistic
        pred = dist.ppf(probs, *params)
        log_qrmse = float(np.sqrt(np.mean((np.log(pred) - np.log(emp)) ** 2)))
        candidates.append({'name': name, 'dist': dist, 'params': params, 'k': k, 'll': ll, 'aic': aic, 'ks': ks, 'log_qrmse': log_qrmse, 'emp_q': emp, 'fit_q': pred})
    return sorted(candidates, key=lambda c: c['aic'])


def plot_hist_model(delta_pct, pos_best, neg_best, weights):
    w_neg, w_zero, w_pos = weights
    width, height = 980, 560
    ml, mr, mt, mb = 78, 24, 28, 64
    x_min, x_max = -180, 600
    bins = np.linspace(x_min, x_max, 95)
    hist, edges = np.histogram(delta_pct[(delta_pct >= x_min) & (delta_pct <= x_max)], bins=bins, density=True)
    x_grid = np.linspace(x_min, x_max, 900)
    y_model = np.zeros_like(x_grid)
    left = x_grid < -100
    right = x_grid > -100
    # If Delta = -100 - 100*M, density is w_neg * f_M((-Delta-100)/100) / 100.
    y_model[left] = w_neg * neg_best['dist'].pdf((-x_grid[left] - 100) / 100, *neg_best['params']) / 100
    # If Delta = -100 + 100*X, density is w_pos * f_X((Delta+100)/100) / 100.
    y_model[right] = w_pos * pos_best['dist'].pdf((x_grid[right] + 100) / 100, *pos_best['params']) / 100
    y_max = max(float(np.max(hist)), float(np.max(y_model[np.isfinite(y_model)]))) * 1.18

    def sx(x): return ml + (x - x_min) / (x_max - x_min) * (width - ml - mr)
    def sy(y): return height - mb - y / y_max * (height - mt - mb)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    # grid and axes
    for xt in [-100, 0, 100, 200, 300, 400, 500, 600]:
        if x_min <= xt <= x_max:
            x = sx(xt)
            parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height-mb}" stroke="#e5e7eb" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{height-36}" text-anchor="middle" font-size="12" fill="#374151">{xt}%</text>')
    for frac in [0, .25, .5, .75, 1.0]:
        yv = frac * y_max
        y = sy(yv)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{yv:.4f}</text>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')

    # histogram bars
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        x = sx(lo); bw = max(1, sx(hi) - sx(lo) - 1); y = sy(h); bh = height - mb - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#93c5fd" opacity="0.58"/>')

    # model line
    pts = []
    for x, y in zip(x_grid, y_model):
        if np.isfinite(y): pts.append(f'{sx(x):.1f},{sy(y):.1f}')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#b91c1c" stroke-width="3"/>')
    # -100 point mass marker
    x_edge = sx(-100)
    parts.append(f'<line x1="{x_edge:.1f}" y1="{mt}" x2="{x_edge:.1f}" y2="{height-mb}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{x_edge+8:.1f}" y="{mt+18}" font-size="12" fill="#5b21b6">atom at -100%: {w_zero:.3%}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">Burn delta percent: empirical histogram vs fitted mixture density</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="13" fill="#111827">BURN_DELTA_MONTHS_PERCENT</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-size="13" fill="#111827">Density</text>')
    parts.append('<rect x="690" y="52" width="250" height="58" fill="white" stroke="#d1d5db"/>')
    parts.append('<rect x="706" y="68" width="26" height="12" fill="#93c5fd" opacity="0.58"/><text x="740" y="79" font-size="12" fill="#111827">Empirical histogram</text>')
    parts.append('<line x1="706" y1="96" x2="732" y2="96" stroke="#b91c1c" stroke-width="3"/><text x="740" y="100" font-size="12" fill="#111827">Fitted mixture density</text>')
    parts.append('</svg>')
    return ''.join(parts)


def plot_qq_positive(pos, pos_best):
    width, height = 740, 560
    ml, mr, mt, mb = 78, 34, 28, 64
    probs = np.linspace(.002, .998, 320)
    emp = np.quantile(pos, probs)
    fit = pos_best['dist'].ppf(probs, *pos_best['params'])
    # log-log makes the tail visible.
    xe = np.log10(fit)
    ye = np.log10(emp)
    lo = min(np.min(xe), np.min(ye)); hi = max(np.max(xe), np.max(ye))
    pad = (hi - lo) * .07
    lo -= pad; hi += pad
    def sx(x): return ml + (x - lo) / (hi - lo) * (width - ml - mr)
    def sy(y): return height - mb - (y - lo) / (hi - lo) * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [-3,-2,-1,0,1]:
        if lo <= t <= hi:
            x=sx(t); y=sy(t)
            parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height-mb}" stroke="#f1f5f9"/>')
            parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#f1f5f9"/>')
            parts.append(f'<text x="{x:.1f}" y="{height-38}" text-anchor="middle" font-size="11" fill="#4b5563">1e{t}</text>')
            parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">1e{t}</text>')
    parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="#111827" stroke-width="1.5"/>')
    for x,y in zip(xe,ye):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2" fill="#2563eb" opacity="0.55"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">Positive component QQ plot: empirical vs Burr XII</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="13" fill="#111827">Fitted positive ratio quantile, log scale</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-size="13" fill="#111827">Empirical positive ratio quantile, log scale</text>')
    parts.append('</svg>')
    return ''.join(parts)


def main():
    ratio, delta_pct, delta_abs, base, burn = load_data()
    pos = ratio[ratio > 0]
    zero = ratio == 0
    neg = ratio[ratio < 0]
    negmag = -neg
    pos_fits = fit_candidates(pos)
    neg_fits = fit_negative(negmag)
    pos_best = pos_fits[0]
    neg_best = neg_fits[0]
    n = len(ratio)
    weights = (len(neg) / n, int(zero.sum()) / n, len(pos) / n)

    quant_probs = [.001,.005,.01,.025,.05,.10,.25,.50,.75,.90,.95,.975,.99,.995,.999]
    quant_rows = [[p, fmt(np.quantile(delta_pct, p), 4), fmt(np.quantile(ratio, p), 6)] for p in quant_probs]
    component_rows = [
        ['Negative correction component', 'R < 0', fmt_int(len(neg)), pct(weights[0]), 'DeltaPct < -100%'],
        ['Zero-burn atom / left edge', 'R = 0', fmt_int(int(zero.sum())), pct(weights[1]), 'DeltaPct = -100%'],
        ['Positive burn continuous component', 'R > 0', fmt_int(len(pos)), pct(weights[2]), 'DeltaPct > -100%'],
    ]
    moment_rows = [
        ['BURN_DELTA_MONTHS_PERCENT', fmt(np.mean(delta_pct), 4), fmt(np.std(delta_pct), 4), fmt(stats.skew(delta_pct), 4), fmt(stats.kurtosis(delta_pct), 4), fmt(np.median(delta_pct), 4)],
        ['R = THIS_BURN / CI_BURN_RATE_MONTHS', fmt(np.mean(ratio), 6), fmt(np.std(ratio), 6), fmt(stats.skew(ratio), 4), fmt(stats.kurtosis(ratio), 4), fmt(np.median(ratio), 6)],
        ['BURN_DELTA_MONTHS', fmt(np.mean(delta_abs), 2), fmt(np.std(delta_abs), 2), fmt(stats.skew(delta_abs), 4), fmt(stats.kurtosis(delta_abs), 4), fmt(np.median(delta_abs), 2)],
    ]
    fit_rows = [[f['name'], tuple(round(float(x), 8) for x in f['params']), fmt(f['ll'], 2), fmt(f['aic'], 2), fmt(f['ks'], 6), fmt(f['log_qrmse'], 6)] for f in pos_fits]
    neg_fit_rows = [[f['name'], tuple(round(float(x), 8) for x in f['params']), fmt(f['ll'], 2), fmt(f['aic'], 2), fmt(f['ks'], 6), fmt(f['log_qrmse'], 6)] for f in neg_fits]
    probs = [.01,.05,.10,.25,.50,.75,.90,.95,.99]
    qfit_rows = []
    for p, emp, fit in zip(probs, pos_best['emp_q'], pos_best['fit_q']):
        qfit_rows.append([p, fmt(emp, 6), fmt(fit, 6), fmt(fit / emp, 4)])

    hist_svg = plot_hist_model(delta_pct, pos_best, neg_best, weights)
    qq_svg = plot_qq_positive(pos, pos_best)

    code = """import csv
from decimal import Decimal

import numpy as np
from scipy import stats

ratio = []
delta_pct = []
with open('ci_payment_details_2.csv', newline='', encoding='utf-8-sig') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        base = float(Decimal(row['CI_BURN_RATE_MONTHS']))
        burn = float(Decimal(row['THIS_BURN']))
        r = burn / base
        ratio.append(r)
        delta_pct.append(100 * (r - 1))

ratio = np.array(ratio)
delta_pct = np.array(delta_pct)
pos = ratio[ratio > 0]
zero_count = np.sum(ratio == 0)
negmag = -ratio[ratio < 0]

pos_params = stats.burr12.fit(pos, floc=0)
neg_params = stats.weibull_min.fit(negmag, floc=0)

print('Positive Burr XII params:', pos_params)
print('Negative magnitude Weibull params:', neg_params)
print('Weights:', {
    'negative': len(negmag) / len(ratio),
    'zero_atom': zero_count / len(ratio),
    'positive': len(pos) / len(ratio),
})"""

    cells = [
        md_cell('# Burn Delta Distribution Analysis\n\nThis notebook characterizes the distribution of burn deltas in `ci_payment_details_2.csv` and proposes a fitted parameterization. The primary modeling variable is the normalized burn ratio `R = THIS_BURN / CI_BURN_RATE_MONTHS`; percent delta is then `100 * (R - 1)`.'),
        md_cell('## Why the Edge Is at -100%\n\nFor `BURN_DELTA_MONTHS_PERCENT`, the left edge for nonnegative burn is `-100%`, not `0`.\n\n```text\nR = THIS_BURN / CI_BURN_RATE_MONTHS\nDeltaPct = 100 * (R - 1)\n```\n\nSo zero actual burn means `R = 0`, which maps to `DeltaPct = -100%`. Negative actual burn creates values below `-100%`. Positive actual burn creates a continuous component above `-100%`.'),
        code_cell(code),
        md_cell('## Empirical Components\n\n' + table(['Component', 'Ratio condition', 'Rows', 'Weight', 'Percent-delta support'], component_rows)),
        md_cell('## Moments and Shape\n\n' + table(['Variable', 'Mean', 'Std dev', 'Skew', 'Excess kurtosis', 'Median'], moment_rows) + '\n\nThe high skew and excess kurtosis rule out a simple normal model. The absolute delta is also strongly scale-dependent, so the normalized ratio or percent delta is the better modeling target.'),
        md_cell('## Quantiles\n\n' + table(['Probability', 'DeltaPct quantile', 'Ratio quantile'], quant_rows)),
        md_cell('## Positive Component Fit\n\nThe continuous positive-burn component is `R > 0`, equivalently `DeltaPct > -100%`. Candidate positive-support distributions were fit by maximum likelihood with `loc=0`.\n\n' + table(['Distribution', 'SciPy params', 'Log likelihood', 'AIC', 'KS statistic', 'Log-quantile RMSE'], fit_rows)),
        md_cell('## Positive Component Quantile Check\n\nBest AIC fit: **' + pos_best['name'] + '**.\n\n' + table(['Probability', 'Empirical positive R', 'Fitted positive R', 'Fit / empirical'], qfit_rows)),
        md_cell('## Negative Correction Component Fit\n\nThe negative component is small, so this fit should be interpreted as a correction/reversal magnitude model rather than a high-confidence physical burn distribution. Let `M = -R` for negative rows.\n\n' + table(['Distribution', 'SciPy params', 'Log likelihood', 'AIC', 'KS statistic', 'Log-quantile RMSE'], neg_fit_rows)),
        md_cell('## Proposed Distribution\n\nModel the normalized burn ratio as a three-part mixture:\n\n```text\nR ~ p_neg * (-M) + p_zero * point_mass(0) + p_pos * X\n```\n\nwith:\n\n```text\np_neg  = ' + f'{weights[0]:.10f}' + '\np_zero = ' + f'{weights[1]:.10f}' + '\np_pos  = ' + f'{weights[2]:.10f}' + '\n\nX ~ BurrXII(c=' + f'{pos_best["params"][0]:.10f}' + ', d=' + f'{pos_best["params"][1]:.10f}' + ', loc=0, scale=' + f'{pos_best["params"][3]:.10f}' + ')\nM ~ Weibull_min(c=' + f'{neg_best["params"][0]:.10f}' + ', loc=0, scale=' + f'{neg_best["params"][2]:.10f}' + ')\n```\n\nTransform to burn-delta percent with:\n\n```text\nDeltaPct = 100 * (R - 1)\n```\n\nEquivalently, in percent-delta space:\n\n```text\nwith probability p_neg:  DeltaPct = -100 - 100*M\nwith probability p_zero: DeltaPct = -100\nwith probability p_pos:  DeltaPct = -100 + 100*X\n```'),
        md_cell('## Plot: Empirical Distribution vs Model\n\nThe histogram is plotted in percent-delta space. The fitted density includes both continuous components; the point mass at `-100%` is shown as a vertical marker because it is an atom, not a density.\n\n<img src="' + svg_data_uri(hist_svg) + '" />'),
        md_cell('## Plot: Positive Component QQ Check\n\nThis compares empirical positive-ratio quantiles to fitted Burr XII quantiles on log scale. Deviations in the far upper tail are expected, but Burr XII captures the body and tail better than the simpler alternatives by AIC.\n\n<img src="' + svg_data_uri(qq_svg) + '" />'),
        md_cell('## Interpretation\n\nThe distribution is best described as a **left-edge-inflated, shifted heavy-tailed mixture**. The edge is `-100%` in percent-delta space because that is where zero actual burn lands. The small mass below `-100%` is a negative correction/reversal process. The large continuous component above `-100%` is positive burn, and it is heavy-tailed: most payment events are below the monthly baseline, but a meaningful tail of events is many times larger than the monthly baseline.\n\nFor modeling absolute deltas, first model `R` or `DeltaPct`, then scale by `CI_BURN_RATE_MONTHS`:\n\n```text\nBURN_DELTA_MONTHS = CI_BURN_RATE_MONTHS * (R - 1)\n```\n\nThis avoids pretending that absolute deltas have one common scale across items.'),
    ]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT_PATH.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print(f'Wrote {OUT_PATH} with {len(cells)} cells')


if __name__ == '__main__':
    main()
