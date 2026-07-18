from __future__ import annotations

import base64
import html
import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

DELAY_CORR = Path('approved_keyword_feature_spearman_correlations_with_sources_2026-06-15.csv')
BUDGET_CORR = Path('project_budget_overrun_approved_feature_correlations_2026-06-15.csv')
DELAY_HEADLINE = Path('project_delay_rf_approved_keywords_headline_comparison_2026-06-11.csv')
DELAY_IMPORT = Path('project_delay_rf_approved_keywords_before_only_binary_top_importances_2026-06-11.csv')
BUDGET_MODEL = Path('project_budget_overrun_rf_approved_keywords_model_results_2026-06-15.csv')
BUDGET_IMPORT = Path('project_budget_overrun_rf_approved_keywords_top_importances_2026-06-15.csv')
SPEND_BUCKETS = Path('project_cumulative_spend_elapsed_bucket_summary_allcustomers.csv')
SPEND_MODEL = Path('project_cumulative_spend_model_family_assessment_allcustomers.csv')
SPEND_RAW = Path('custpaydetails_project_clustered_cumulative_curves_allcustomers_2026-06-08-1000.csv')
KW_DETAIL = Path('approved_keyword_feature_column_filter_detail_2026-06-11.csv')
CO_TIME_BIN = Path('change_order_time_lifecycle_bin_summary_2026-06-14.csv')
CO_BUDGET_BIN = Path('change_order_budget_lifecycle_bin_summary_2026-06-14.csv')
CO_LIFT = Path('project_delay_50pct_co_lift_vs_baseline_2026-06-14.csv')
CO_SPARSITY = Path('project_delay_50pct_co_feature_sparsity_diagnostics_2026-06-14.csv')
FEATURE_DATA = Path('custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv')
OUT = Path('project_risk_model_executive_summary_2026-06-15.html')

INK = '#172033'
MUTED = '#667085'
BLUE = '#2563eb'
TEAL = '#0f766e'
ORANGE = '#f97316'
RED = '#dc2626'
GREEN = '#16a34a'
GRID = '#eef2f6'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'axes.edgecolor': '#d0d5dd',
    'axes.labelcolor': INK,
    'xtick.color': '#344054',
    'ytick.color': '#344054',
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})


def style(ax):
    ax.grid(True, axis='y', color=GRID, lw=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#d0d5dd')


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    svg = buf.getvalue()
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg).decode('ascii')


def short(s, n=42):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + '...'


def top_corr_table(corr: pd.DataFrame, direction: str, n=20) -> str:
    d = corr[corr.direction.eq(direction)].head(n).copy()
    d['rank'] = d['direction_rank'].astype('Int64')
    d['feature'] = d['feature'].map(lambda x: short(x, 44))
    d['spearman_r'] = d['spearman_r'].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    d = d[['rank','feature','spearman_r','class']]
    return dataframe_to_html(d, ['Rank','Feature','Spearman r','Class'])


def dataframe_to_html(df: pd.DataFrame, headers=None, classes='data-table') -> str:
    heads = headers or list(df.columns)
    out = [f'<table class="{classes}"><thead><tr>']
    for h in heads:
        out.append(f'<th>{html.escape(str(h))}</th>')
    out.append('</tr></thead><tbody>')
    for _, row in df.iterrows():
        out.append('<tr>')
        for v in row.tolist():
            if pd.isna(v):
                txt = ''
            elif isinstance(v, (float, np.floating)):
                txt = f'{v:.3f}'
            else:
                txt = str(v)
            out.append(f'<td>{html.escape(txt)}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def metric_card(label, value, detail=''):
    return f'<div class="metric"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div><div class="metric-detail">{html.escape(detail)}</div></div>'


def chart_cumulative_spend():
    sb = pd.read_csv(SPEND_BUCKETS)
    sm = pd.read_csv(SPEND_MODEL)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7), gridspec_kw={'width_ratios':[1.55,1], 'wspace':0.58})
    ax = axes[0]
    x = sb.x_mid * 100
    ax.fill_between(x, sb.p10*100, sb.p90*100, color='#bfdbfe', alpha=.55, label='10th-90th percentile')
    ax.fill_between(x, sb.p25*100, sb.p75*100, color='#60a5fa', alpha=.35, label='25th-75th percentile')
    ax.plot(x, sb['median']*100, color=BLUE, lw=2.6, label='Median')
    ax.plot([0,100],[0,100], color='#98a2b3', ls='--', label='Linear')
    ax.set_xlabel('Elapsed project life (%)')
    ax.set_ylabel('Cumulative spend (%)')
    ax.set_title('Empirical cumulative spend bands')
    ax.legend(fontsize=8, loc='lower right')
    style(ax)
    ax2 = axes[1]
    mm = sm.sort_values('MAE')
    labels = [short(x, 28) for x in mm.ModelFamily]
    ax2.barh(labels, mm.MAE, color=[TEAL if i==0 else '#94a3b8' for i in range(len(mm))])
    ax2.invert_yaxis()
    ax2.set_xlabel('MAE')
    ax2.tick_params(axis='y', labelsize=8, pad=2)
    ax2.set_title('Model family screen')
    style(ax2)
    return fig_to_data_uri(fig)


def chart_duration():
    raw = pd.read_csv(SPEND_RAW, low_memory=False)
    raw['duration_bucket'] = pd.cut(raw.PROJECTMODELEDDAYS, [-np.inf,60,180,365,np.inf], labels=['<60 days','60-180 days','181-365 days','>365 days'])
    raw['elapsed_bin'] = pd.cut(raw.ELAPSEDPCT, np.linspace(0,1,11), include_lowest=True)
    dur = raw.groupby(['duration_bucket','elapsed_bin'], observed=True).agg(x=('ELAPSEDPCT','median'), y=('CUMULATIVEBURNPCT','median')).reset_index()
    fig, ax = plt.subplots(figsize=(9.3,4.8))
    colors = {'<60 days':RED,'60-180 days':ORANGE,'181-365 days':BLUE,'>365 days':TEAL}
    for b, g in dur.groupby('duration_bucket', observed=True):
        ax.plot(g.x*100, g.y*100, marker='o', lw=2.2, label=str(b), color=colors[str(b)])
    ax.plot([0,100],[0,100], color='#98a2b3', ls='--')
    ax.set_xlabel('Elapsed project life (%)')
    ax.set_ylabel('Median cumulative spend (%)')
    ax.set_title('Median spend curve by project duration')
    ax.legend(fontsize=9)
    style(ax)
    return fig_to_data_uri(fig)


def chart_keyword_filter():
    kw = pd.read_csv(KW_DETAIL)
    fig, axes = plt.subplots(1, 2, figsize=(12,4.3))
    stages = ['Original\nkeyword groups','After frequency +\nmulti-customer filter','After manual\nsemantic review']
    vals = [3000,2069,1940]
    axes[0].bar(stages, vals, color=[BLUE, TEAL, GREEN])
    axes[0].set_ylabel('Keyword family groups')
    axes[0].set_title('Keyword filtering funnel')
    for i, v in enumerate(vals):
        axes[0].text(i, v+45, f'{v:,}', ha='center', fontsize=10, weight='bold')
    style(axes[0])
    fam = kw.groupby(['family','retained_in_output']).size().reset_index(name='count')
    ret = fam[fam.retained_in_output == True].set_index('family')['count']
    drop = fam[fam.retained_in_output == False].set_index('family')['count']
    labs = ['project','contract','item']
    x = np.arange(len(labs))
    axes[1].bar(x, [ret.get(l,0) for l in labs], label='retained', color=GREEN)
    axes[1].bar(x, [drop.get(l,0) for l in labs], bottom=[ret.get(l,0) for l in labs], label='dropped', color='#fca5a5')
    axes[1].set_xticks(x, labs)
    axes[1].set_ylabel('Feature columns')
    axes[1].set_title('Retained vs dropped feature columns')
    axes[1].legend(fontsize=8)
    style(axes[1])
    return fig_to_data_uri(fig)


def chart_delay_model():
    dhead = pd.read_csv(DELAY_HEADLINE)
    hh = dhead[dhead.FeatureSet.eq('before_only')].copy()
    hh['Label'] = hh.Split.str.replace('Hash 80/20','Hash holdout').str.replace('Time old 80/new 20','Time holdout')
    fig, axes = plt.subplots(1, 2, figsize=(12,4.4), gridspec_kw={'width_ratios':[1.55,1]})
    ax = axes[0]
    idx = np.arange(len(hh)); w = .25
    ax.bar(idx-w, hh.Binary_AUC, w, label='Binary AUC', color=BLUE)
    ax.bar(idx, hh.ThreeBin_BalancedAccuracy, w, label='Three-bin balanced accuracy', color=TEAL)
    ax.bar(idx+w, hh.Regression_R2, w, label='Regression R2', color=ORANGE)
    ax.set_xticks(idx, hh.Label)
    ax.set_ylim(0,1.05)
    ax.set_title('Delay model summary metrics')
    ax.legend(fontsize=8)
    style(ax)
    ax2 = axes[1]
    ax2.bar(['MAE','RMSE'], [hh.iloc[0].Regression_MAE, hh.iloc[0].Regression_RMSE], color=[BLUE, TEAL])
    ax2.set_title('Hash holdout regression error')
    ax2.set_ylabel('PercentDelayed points')
    style(ax2)
    return fig_to_data_uri(fig)


def chart_budget_model():
    bm = pd.read_csv(BUDGET_MODEL)
    breg = bm[bm.Task.eq('regression')].iloc[0]
    bbin = bm[bm.Task.eq('binary')].iloc[0]
    btri = bm[bm.Task.eq('three_bin')].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12,4.4))
    metrics = ['Binary AUC','Binary bal. acc.','Three-bin bal. acc.','Three-bin OVR AUC']
    vals = [bbin.AUC, bbin.BalancedAccuracy, btri.BalancedAccuracy, btri.AUC]
    axes[0].barh(metrics, vals, color=[BLUE, TEAL, ORANGE, GREEN])
    axes[0].set_xlim(0,1)
    axes[0].set_title('Budget classification metrics')
    style(axes[0])
    axes[1].bar(['MAE','RMSE'], [breg.MAE, breg.RMSE], color=[BLUE, TEAL])
    axes[1].set_ylabel('Budget-overrun percentage points')
    axes[1].set_title('Budget regression error')
    style(axes[1])
    return fig_to_data_uri(fig)


def chart_importances():
    di = pd.read_csv(DELAY_IMPORT).head(15)
    bi = pd.read_csv(BUDGET_IMPORT).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(12.5,5.2))
    axes[0].barh([short(x,30) for x in di.Feature[::-1]], di.Importance[::-1], color=BLUE)
    axes[0].set_title('Delay binary model')
    style(axes[0])
    axes[1].barh([short(x,30) for x in bi.Feature[::-1]], bi.Importance[::-1], color=TEAL)
    axes[1].set_title('Budget binary model')
    style(axes[1])
    fig.tight_layout()
    return fig_to_data_uri(fig)


def chart_change_orders():
    ct = pd.read_csv(CO_TIME_BIN)
    cb = pd.read_csv(CO_BUDGET_BIN)
    fig, axes = plt.subplots(1, 2, figsize=(12.5,4.5))
    tb = ct[ct.TimingBasis.eq('ApprovedOn')]
    axes[0].bar(tb.LifecycleBin, tb.TotalAddedDays, color=ORANGE)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_title('Time COs: added days by approval timing')
    axes[0].set_ylabel('Total added days')
    style(axes[0])
    bb = cb[cb.TimingBasis.eq('ApprovedOn')]
    axes[1].bar(bb.LifecycleBin, bb.NetAmountDelta/1e6, color=TEAL)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('Budget COs: net $ delta by approval timing')
    axes[1].set_ylabel('Net amount delta ($M)')
    style(axes[1])
    fig.tight_layout()
    return fig_to_data_uri(fig)


def chart_co_diagnostics():
    cs = pd.read_csv(CO_SPARSITY).head(10)
    cl = pd.read_csv(CO_LIFT)
    fig, axes = plt.subplots(1, 2, figsize=(12.5,4.8))
    axes[0].barh([short(x,35) for x in cs.Feature[::-1]], cs.NonZeroShare[::-1]*100, color=BLUE)
    axes[0].set_xlabel('Projects with nonzero feature (%)')
    axes[0].set_title('Early CO feature sparsity')
    style(axes[0])
    lift = cl[cl.Split.eq('hash_80_20')]
    axes[1].barh(lift.FeatureSet, lift.BalancedAccuracyLiftVsBaseline, color=[GREEN if v>0 else RED for v in lift.BalancedAccuracyLiftVsBaseline])
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_title('Lift vs beginning-only baseline')
    axes[1].set_xlabel('Balanced accuracy lift')
    style(axes[1])
    fig.tight_layout()
    return fig_to_data_uri(fig)


def build_html():
    delay_corr = pd.read_csv(DELAY_CORR)
    budget_corr = pd.read_csv(BUDGET_CORR)
    dhead = pd.read_csv(DELAY_HEADLINE)
    bm = pd.read_csv(BUDGET_MODEL)
    breg = bm[bm.Task.eq('regression')].iloc[0]
    bbin = bm[bm.Task.eq('binary')].iloc[0]
    btri = bm[bm.Task.eq('three_bin')].iloc[0]
    dhash = dhead[(dhead.FeatureSet.eq('before_only')) & (dhead.Split.eq('Hash 80/20'))].iloc[0]
    data_rows = sum(1 for _ in open(FEATURE_DATA, 'rb')) - 1

    css = '''
:root { --ink:#172033; --muted:#667085; --line:#e4e7ec; --bg:#f7f9fc; --blue:#2563eb; --teal:#0f766e; --orange:#f97316; --green:#16a34a; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, Arial, Helvetica, sans-serif; color:var(--ink); background:#fff; line-height:1.45; }
.report { max-width: 1120px; margin: 0 auto; padding: 44px 42px 72px; }
.cover { min-height: 760px; display:flex; flex-direction:column; justify-content:center; border-bottom: 1px solid var(--line); }
.eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.08em; font-size:12px; font-weight:700; }
h1 { font-size:54px; line-height:1.02; margin:16px 0 18px; max-width:850px; }
.subtitle { font-size:21px; color:var(--muted); max-width:850px; }
.meta { margin-top:42px; color:var(--muted); font-size:14px; }
section { padding: 42px 0; border-bottom:1px solid var(--line); break-inside: avoid; }
h2 { font-size:28px; line-height:1.15; margin:0 0 10px; }
.lede { color:var(--muted); font-size:16px; max-width:880px; margin:0 0 22px; }
.grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:22px; align-items:start; }
.grid-3 { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:16px; }
.card { border:1px solid var(--line); border-radius:8px; background:#fff; padding:18px; }
.callout { border-left:4px solid var(--blue); background:#f8fbff; padding:16px 18px; border-radius:6px; margin:18px 0; }
.chart { width:100%; display:block; }
.metric { border:1px solid var(--line); border-radius:8px; padding:16px; background:#fcfcfd; }
.metric-label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-weight:700; }
.metric-value { font-size:28px; margin-top:4px; font-weight:800; color:var(--ink); }
.metric-detail { font-size:12px; color:var(--muted); margin-top:4px; }
ul { margin:12px 0 0 20px; padding:0; }
li { margin:7px 0; }
.data-table { width:100%; border-collapse:collapse; font-size:11px; table-layout:fixed; }
.data-table th { background:#f2f4f7; color:#344054; text-align:left; padding:7px; border:1px solid var(--line); font-size:10px; }
.data-table td { padding:6px 7px; border:1px solid #eef2f6; vertical-align:top; word-break:break-word; }
.data-table tr:nth-child(even) td { background:#fcfcfd; }
.appendix-note { font-size:12px; color:var(--muted); }
@media print { body { background:#fff; } .report { max-width:none; padding:24px 28px; } section { break-inside: avoid; page-break-inside: avoid; } .cover { page-break-after:always; } .page-break { page-break-before:always; } h2 { break-after:avoid; } .data-table { font-size:9px; } .grid { gap:14px; } }
'''

    html_doc = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project Risk Modeling Executive Summary</title><style>{css}</style></head>
<body><main class="report">
<div class="cover">
  <div class="eyebrow">Executive Summary</div>
  <h1>Project Risk Modeling</h1>
  <p class="subtitle">Cumulative spend behavior, delay risk, budget-overrun risk, and the current value of change-order data.</p>
  <p class="meta">Prepared from project-level aggregate data across all customers<br>Generated 2026-06-15</p>
</div>

<section>
  <h2>Overall Findings</h2>
  <p class="lede">The analysis supports project-level aggregate decision making as the right framing for time-delay and budget-overrun risk. Contract items explain low-level burn behavior, but operational risk decisions are made at the project level, where many items with different timing and value combine into one lifecycle.</p>
  <div class="grid-3">
    {metric_card('Approved feature rows', f'{data_rows:,}', 'Project-level records in approved dataset')}
    {metric_card('Delay binary AUC', f'{dhash.Binary_AUC:.3f}', 'Before-only approved feature model')}
    {metric_card('Budget binary AUC', f'{bbin.AUC:.3f}', 'New budget-overrun POC model')}
  </div>
  <div class="callout"><strong>Model positioning:</strong> these are proof-of-concept risk screens, not final production models. Their value is showing that the data contains useful signal and clarifying which families of information appear informative.</div>
</section>

<section>
  <h2>Project Cumulative Spend Is S-Shaped</h2>
  <p class="lede">At the project level, cumulative spend is not linear. The median curve rises slowly early, steepens through the middle of the lifecycle, and then flattens toward completion.</p>
  <img class="chart" src="{chart_cumulative_spend()}" alt="Cumulative spend bands and model family screen">
  <ul><li>The linear reference underestimates spend through much of the middle lifecycle.</li><li>Empirical/isotonic curves and S-curve families are reasonable baselines for aggregate reporting.</li><li>The wide percentile bands show substantial project-to-project variation, so any production benchmark should include uncertainty bands.</li></ul>
</section>

<section>
  <h2>Duration Changes The Curve Shape</h2>
  <p class="lede">The S-shape is clearest for mid-duration projects. Short projects under 60 days often lack enough lifecycle to show the curve; long projects over 365 days compress a fixed 2-3 week ramp-up into a small part of total project life.</p>
  <div class="grid"><div><img class="chart" src="{chart_duration()}" alt="Duration stratified curves"></div><div class="card"><h3>Interpretation</h3><ul><li>Short projects are dominated by payment bursts rather than stable progression.</li><li>Mid-duration projects show the clearest nonlinear lifecycle pattern.</li><li>Long projects show an apparently truncated early ramp, consistent with a ramp-up period that does not expand proportionally with duration.</li><li>Duration should be a core stratification variable for production expected-spend curves.</li></ul></div></div>
</section>

<section>
  <h2>Feature Generation And Keyword Filtering</h2>
  <p class="lede">The feature table combines mathematical transformations of numeric project setup fields with keyword features extracted from project, contract, and item text. Keywords were filtered to reduce sparse, customer-specific, and semantically weak signals.</p>
  <img class="chart" src="{chart_keyword_filter()}" alt="Keyword filtering funnel">
  <ul><li>Original keyword candidates: 3,000 keyword groups and 6,000 keyword-derived feature columns.</li><li>Automated screen: retain only keyword families appearing in at least 4 projects and more than 1 customer.</li><li>Manual review: remove generic administrative terms, geography/place proxies, mixed-family noise, and weak abbreviations.</li><li>Final approved dataset: 75 non-keyword columns plus 4,141 approved keyword feature columns.</li></ul>
</section>

<section class="page-break">
  <h2>Top Correlations With Schedule Delay</h2>
  <p class="lede">Spearman correlation measures monotonic association with project delay. Positive means larger feature values tend to align with greater delay; negative means larger values tend to align with less delay.</p>
  <div class="grid"><div class="card"><h3>Top Positive</h3>{top_corr_table(delay_corr, 'positive')}</div><div class="card"><h3>Top Negative</h3>{top_corr_table(delay_corr, 'negative')}</div></div>
</section>

<section class="page-break">
  <h2>Top Correlations With Budget Overrun</h2>
  <p class="lede">Budget overrun is defined as posted project work completed divided by planned project value, minus 100%. This is a new report-level target and should be validated as a final actual-cost proxy.</p>
  <div class="grid"><div class="card"><h3>Top Positive</h3>{top_corr_table(budget_corr, 'positive')}</div><div class="card"><h3>Top Negative</h3>{top_corr_table(budget_corr, 'negative')}</div></div>
</section>

<section>
  <h2>Proof-of-Concept Delay Model</h2>
  <p class="lede">The regularized random forest delay model is strongest as a binary delayed/not-delayed risk screen. Severity prediction is harder because exact delay magnitude depends on later execution conditions.</p>
  <img class="chart" src="{chart_delay_model()}" alt="Delay model performance">
</section>

<section>
  <h2>Proof-of-Concept Budget-Overrun Model</h2>
  <p class="lede">The budget-overrun model uses the same approved beginning-available feature set and the new budget target definition. It is useful as an initial signal screen, not a final cost model.</p>
  <div class="grid-3">
    {metric_card('Budget binary AUC', f'{bbin.AUC:.3f}', 'Ranking overrun vs non-overrun')}
    {metric_card('Three-bin bal. accuracy', f'{btri.BalancedAccuracy:.3f}', 'On/under, mild overrun, severe overrun')}
    {metric_card('Regression MAE', f'{breg.MAE:.2f}', 'Budget-overrun percentage points')}
  </div>
  <img class="chart" src="{chart_budget_model()}" alt="Budget model performance">
</section>

<section>
  <h2>What The Models Rely On</h2>
  <p class="lede">Random forest importance is not causal evidence, but it is a useful sanity check. The leading drivers are generally project scale, schedule intensity, item price distributions, standard item diversity, and selected scope keywords.</p>
  <img class="chart" src="{chart_importances()}" alt="Model feature importances">
</section>

<section>
  <h2>Change Orders: Timing In The Lifecycle</h2>
  <p class="lede">Change orders are important project controls data, but many occur late in the planned project lifecycle. Values above 100% are intentionally not clipped because they represent activity after the original planned end date.</p>
  <img class="chart" src="{chart_change_orders()}" alt="Change order lifecycle timing">
</section>

<section>
  <h2>Why Early Change Orders Did Not Improve The Delay Model</h2>
  <p class="lede">At 50% of planned project life, early approved change-order features are sparse and unstable. The current sample does not support relying on them as the primary early prediction signal.</p>
  <img class="chart" src="{chart_co_diagnostics()}" alt="Change order diagnostics">
  <ul><li>Early time change orders are especially sparse.</li><li>CO-only models showed weak standalone predictive power for three-bin delay classification.</li><li>Regularized forests prefer broad setup features because they help many more project splits.</li><li>Change orders remain valuable for monitoring and escalation once they appear.</li></ul>
</section>

<section>
  <h2>Limitations And Recommended Next Steps</h2>
  <div class="grid"><div class="card"><h3>Limitations</h3><ul><li>Correlation and feature importance do not establish causality.</li><li>The budget target depends on posted work completed amount being a reliable actual-cost proxy.</li><li>Some planned-date and budget-linkage fields need client validation for beginning availability.</li><li>Customer-specific caveats are intentionally omitted from this executive view.</li></ul></div><div class="card"><h3>Next Steps</h3><ul><li>Get sign-off from Aurigo that the proposed features are allowed because they are expected to be present at the beginning of the project.</li><li>Investigate integrating canonical phrases from Construction Specifications.</li><li>Build out a more sophisticated model.</li></ul></div></div>
  <p class="appendix-note">Generated artifacts: project_budget_overrun_approved_feature_correlations_2026-06-15.csv; project_budget_overrun_rf_approved_keywords_model_results_2026-06-15.csv; project_budget_overrun_rf_approved_keywords_top_importances_2026-06-15.csv.</p>
</section>
</main></body></html>'''
    OUT.write_text(html_doc, encoding='utf-8')


if __name__ == '__main__':
    build_html()
    print('Wrote', OUT)
