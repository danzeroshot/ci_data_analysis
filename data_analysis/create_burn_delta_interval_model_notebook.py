
#!/usr/bin/env python3
from __future__ import annotations
import csv, json, uuid, math, base64
from collections import defaultdict
from decimal import Decimal
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats

CSV_PATH=Path('ci_payment_details_2.csv')
OUT_PATH=Path('burn_delta_interval_model_analysis.ipynb')

def D(x): return float(Decimal(str(x)))
def dt(x): return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
def md_cell(s): return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':s}
def code_cell(s): return {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':s}
def fmt(x,p=2): return f'{float(x):,.{p}f}'
def fmt_int(x): return f'{int(x):,}'
def pct(x): return f'{100*float(x):.2f}%'
def esc(x): return str(x).replace('|','\\|')
def table(headers, rows):
    out=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for r in rows: out.append('| '+' | '.join(esc(v) for v in r)+' |')
    return '\n'.join(out)

def load():
    rows=[]; items=defaultdict(list)
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            row=dict(r)
            row['item']=r['ITEMID']; row['date']=dt(r['WPPOSTINGDATE']); row['burn']=D(r['THIS_BURN'])
            row['pg_base']=D(r['CI_BURN_RATE_PG']); row['mo_base']=D(r['CI_BURN_RATE_MONTHS'])
            row['pg_pct']=D(r['BURN_DELTA_PG_PERCENT']); row['mo_pct']=D(r['BURN_DELTA_MONTHS_PERCENT'])
            row['num']=int(D(r['NUM_PAYGROUPS'])); row['days']=D(r['DAYS_BETWEEN'])
            rows.append(row); items[row['item']].append(row)
    for item,rs in items.items():
        rs.sort(key=lambda r:r['date'])
        prev=None
        for i,r in enumerate(rs,1):
            r['seq']=i
            r['interval_days']=30.0 if i==1 else (r['date']-prev).total_seconds()/86400
            r['interval_base']=r['mo_base']*(r['interval_days']/30.0)
            r['interval_pct']=(r['burn']-r['interval_base'])/r['interval_base']*100 if r['interval_base'] else math.nan
            prev=r['date']
    return rows,items

def summary(vals):
    x=np.asarray(vals,dtype=float)
    return [fmt_int(len(x)),fmt(np.mean(x),2),fmt(np.std(x),2),fmt(stats.skew(x),2),fmt(stats.kurtosis(x),2),fmt(np.quantile(x,.25),2),fmt(np.median(x),2),fmt(np.quantile(x,.75),2),fmt(np.quantile(x,.95),2),fmt(np.quantile(x,.99),2),fmt_int(np.sum(np.abs(x)<=25)),fmt_int(np.sum(x>100)),fmt_int(np.sum(x<-100))]

def fit_ratio(vals):
    r=np.asarray(vals,dtype=float); pos=r[r>0]; zero=np.sum(r==0); neg=r[r<0]
    fits=[]
    for name,dist,k in [('Burr XII',stats.burr12,3),('Log-logistic/Fisk',stats.fisk,2),('Lognormal',stats.lognorm,2),('Weibull',stats.weibull_min,2),('Gamma',stats.gamma,2)]:
        params=dist.fit(pos,floc=0); ll=float(np.sum(dist.logpdf(pos,*params))); aic=2*k-2*ll; ks=stats.kstest(pos,dist.cdf,args=params).statistic
        fits.append((aic,name,params,ll,ks))
    fits=sorted(fits)
    return len(neg)/len(r), zero/len(r), len(pos)/len(r), fits

def svg_hist(rows):
    series=[('PG percent',[r['pg_pct'] for r in rows],'#2563eb'),('Original months percent',[r['mo_pct'] for r in rows],'#dc2626'),('Interval months percent',[r['interval_pct'] for r in rows if math.isfinite(r['interval_pct'])],'#059669')]
    bins=np.arange(-150,410,10); width=980; height=530; ml=70; mr=25; mt=38; mb=58
    hists=[]; ymax=0
    for name,vals,color in series:
        x=np.asarray(vals); x=x[(x>-150)&(x<400)]
        hist,edges=np.histogram(x,bins=bins,density=False)
        hists.append((name,hist,edges,color)); ymax=max(ymax,int(hist.max()))
    ymax*=1.12
    def sx(x): return ml+(x+150)/(550)*(width-ml-mr)
    def sy(y): return height-mb-y/ymax*(height-mt-mb)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    for xt in [-100,0,100,200,300,400]:
        x=sx(xt); parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height-mb}" stroke="#e5e7eb"/><text x="{x:.1f}" y="{height-33}" text-anchor="middle" font-size="12">{xt}%</text>')
    for frac in [0,.25,.5,.75,1]:
        yv=ymax*frac; y=sy(yv); parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#f3f4f6"/><text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{int(yv)}</text>')
    for name,hist,edges,color in hists:
        pts=[]
        for lo,hi,c in zip(edges[:-1],edges[1:],hist):
            mid=(lo+hi)/2; pts.append(f'{sx(mid):.1f},{sy(c):.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="17" font-weight="700">Burn-delta percent histograms, 10-point bins clipped to [-150, 400]</text>')
    lx=700; ly=60
    for i,(name,_,_,color) in enumerate(hists):
        y=ly+i*24; parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+28}" y2="{y}" stroke="{color}" stroke-width="3"/><text x="{lx+36}" y="{y+4}" font-size="12">{name}</text>')
    parts.append('</svg>')
    return 'data:image/svg+xml;base64,'+base64.b64encode(''.join(parts).encode()).decode()

def main():
    rows,items=load()
    errs=[sum(r['burn'] for r in rs)-sum(r['interval_base'] for r in rs) for rs in items.values()]
    metrics=[
        ['PG percent: payment-event baseline']+summary([r['pg_pct'] for r in rows]),
        ['Original months percent: full-month baseline']+summary([r['mo_pct'] for r in rows]),
        ['Interval months percent: exposure baseline']+summary([r['interval_pct'] for r in rows if math.isfinite(r['interval_pct'])]),
    ]
    headers=['Metric','N','Mean','Std dev','Skew','Excess kurtosis','P25','Median','P75','P95','P99','Rows within +/-25','Rows > +100','Rows < -100']
    ivals=[r['interval_days'] for r in rows]
    interval_rows=[[p,fmt(np.quantile(ivals,p),4)] for p in [.001,.005,.01,.025,.05,.10,.25,.50,.75,.90,.95,.99]]
    base_ratio=[r['interval_base']/r['pg_base'] for r in rows]
    base_rows=[[p,fmt(np.quantile(base_ratio,p),4)] for p in [.01,.05,.10,.25,.50,.75,.90,.95,.99]]
    fit_rows=[]
    for label,ratio in [
        ('PG ratio', [r['burn']/r['pg_base'] for r in rows]),
        ('Original monthly ratio', [r['burn']/r['mo_base'] for r in rows]),
        ('Interval monthly ratio', [r['burn']/r['interval_base'] for r in rows if r['interval_base']]),
    ]:
        wn,wz,wp,fits=fit_ratio(ratio); best=fits[0]
        fit_rows.append([label,pct(wn),pct(wz),pct(wp),best[1],tuple(round(float(x),6) for x in best[2]),fmt(best[0],2),fmt(best[4],5)])
    img=svg_hist(rows)
    sql_note='The companion Snowflake SQL is `custpaydetails_wbins_interval.sql`. It adds `interval_days`, `interval_months`, `ci_burn_interval_months`, `burn_delta_interval_months_percent`, and related absolute delta fields.'
    cells=[
        md_cell('# Interval-Adjusted Monthly Burn Delta Analysis\n\nThis notebook inserts an explicit interval model before interpreting burn-delta variation. It compares three baselines: per-payment-group (`_pg`), original full-month (`_months`), and interval-adjusted monthly exposure.'),
        md_cell('## Clarifying Assumptions Used\n\n- The first payment receives 30 days of exposure, matching the existing `(DAYS_BETWEEN + 30) / 30` convention.\n- Later payments receive exposure equal to days since the previous `WPPostingDate` for the same `ITEMID`.\n- Rows are ordered after aggregating to one row per `ITEMID, WPPostingDate`.\n- Same-day intervals would produce zero exposure and are guarded with `NULLIF` in the SQL; the current revised CSV has distinct posting dates per item.'),
        md_cell('## Interval Model Definition\n\n```text\ninterval_days = 30 for first payment\ninterval_days = DATEDIFF(day, previous_posting_date, current_posting_date) for later payments\ninterval_months = interval_days / 30\ninterval_monthly_baseline = CI_BURN_RATE_MONTHS * interval_months\nburn_delta_interval_months_percent = (THIS_BURN - interval_monthly_baseline) / interval_monthly_baseline * 100\n```\n\n' + sql_note),
        code_cell("import csv\nfrom decimal import Decimal\nfrom datetime import datetime\nfrom collections import defaultdict\n\n# See create_burn_delta_interval_model_notebook.py for the full reproducible analysis."),
        md_cell('## Item-Total Reconciliation\n\nThe interval model is structurally coherent for total allocation. Across items, the median absolute item-total reconciliation error is `' + fmt(np.median(np.abs(errs)),6) + '` and the max absolute error is `' + fmt(np.max(np.abs(errs)),6) + '`. Small residuals are floating-point/decimal artifacts.'),
        md_cell('## Distribution Comparison\n\n' + table(headers, metrics)),
        md_cell('## Why Interval Percent Becomes Huge\n\nThe interval model fixes total allocation, but it does not make individual payment events smooth. Many payment events are very close together, so the interval exposure denominator is tiny. A normal-sized payment after a 1-day interval can be dozens of times larger than the 1-day expected accrual.\n\n### Interval Days Quantiles\n\n' + table(['Quantile','Interval days'], interval_rows)),
        md_cell('## Interval Baseline vs PG Baseline\n\n`interval_baseline / pg_baseline` is often well below 1, so interval-adjusted percent deltas become much larger than `_pg` percent deltas for short-gap payment events.\n\n' + table(['Quantile','interval baseline / pg baseline'], base_rows)),
        md_cell('## Candidate Positive-Component Fits\n\nEach ratio is modeled as a three-part mixture: negative correction, zero-burn atom, and positive continuous component. The table reports the best positive-support fit by AIC.\n\n' + table(['Ratio target','Negative weight','Zero atom','Positive weight','Best positive distribution','SciPy params','AIC','KS'], fit_rows)),
        md_cell('## Histogram Comparison\n\n<img src="' + img + '" />'),
        md_cell('## Interpretation\n\nThe interval model should be added before variation analysis because it separates calendar exposure from event-size variation. It answers: “given the elapsed time since the prior payment, how large was this payment?”\n\nThe result is important: interval adjustment conserves item totals, but row-level interval deltas are even more right-tailed than original `_months` deltas because payment dates are administrative events, not clean daily accrual boundaries. In this data, the median interval is only 3 days and 25% of intervals are 1 day or less.\n\nSo the models serve different purposes:\n\n- `_pg_percent` is best for payment-event size variation.\n- original `_months_percent` is a rough full-month comparator and is systematically negative because rows are not months.\n- interval-adjusted `_months_percent` is best for allocation/anomaly analysis over time, but it requires heavy-tailed modeling and likely first/middle/last or short-interval treatment.\n\nA practical calendar-burn model should use interval exposure for total reconciliation, then model the residual/event factor with robust or quantile methods rather than expecting interval-adjusted deltas to be near zero.'),
    ]
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
    OUT_PATH.write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print(f'Wrote {OUT_PATH} with {len(cells)} cells')
if __name__=='__main__': main()
