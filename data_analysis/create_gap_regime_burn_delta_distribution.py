
#!/usr/bin/env python3
from __future__ import annotations
import csv, json, uuid, math
from collections import defaultdict
from decimal import Decimal
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats

IN=Path('ci_payment_details_2.csv')
OUTCSV=Path('ci_payment_details_2_gap_regimes.csv')
OUTNB=Path('gap_regime_burn_delta_distribution.ipynb')

def D(x): return float(Decimal(str(x)))
def dt(x): return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
def regime(is_first,gap):
    if is_first: return '00_first_payment'
    if gap <= 0.5: return '01_intraday_or_same_day'
    if gap <= 3: return '02_1_to_3_day_burst'
    if gap <= 14: return '03_4_to_14_day_short_cycle'
    if gap <= 20: return '04_15_to_20_day_transition'
    if gap <= 45: return '05_21_to_45_day_monthlyish'
    if gap <= 59: return '06_46_to_59_day_transition'
    return '07_60_plus_day_long_gap'
def md(s): return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':s}
def code(s): return {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':s}
def table(headers,rows):
    out=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for r in rows: out.append('| '+' | '.join(str(x).replace('|','\\|') for x in r)+' |')
    return '\n'.join(out)
def fmt(x,p=2): return f'{float(x):,.{p}f}'
def fmt_int(x): return f'{int(x):,}'
def pct(x): return f'{100*float(x):.2f}%'
def summ(vals):
    x=np.array([v for v in vals if math.isfinite(v)])
    return [fmt_int(len(x)),fmt(np.median(x),2),fmt(np.quantile(x,.25),2),fmt(np.quantile(x,.75),2),fmt(np.quantile(x,.95),2),fmt(np.quantile(x,.99),2),pct(np.mean(np.abs(x)<=25)),pct(np.mean(x>100)),pct(np.mean(x<-100))]

def main():
    rows=[]; items=defaultdict(list)
    with IN.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        base_fields=reader.fieldnames
        for r in reader:
            row=dict(r); row['_date']=dt(r['WPPOSTINGDATE']); row['_item']=r['ITEMID']
            rows.append(row); items[row['_item']].append(row)
    enriched=[]
    for item,rs in items.items():
        rs.sort(key=lambda r:r['_date'])
        prev=None
        for i,r in enumerate(rs,1):
            first=i==1
            gap=30.0 if first else (r['_date']-prev).total_seconds()/86400
            burn=D(r['THIS_BURN']); pg=D(r['CI_BURN_RATE_PG']); mo=D(r['CI_BURN_RATE_MONTHS'])
            interval_base=mo*(gap/30.0)
            out={k:v for k,v in r.items() if not k.startswith('_')}
            out['PAYMENT_SEQUENCE']=i
            out['IS_FIRST_PAYMENT']='1' if first else '0'
            out['GAP_DAYS_SINCE_PRIOR']=f'{gap:.10f}'
            out['GAP_REGIME']=regime(first,gap)
            out['INTERVAL_MONTHS']=f'{gap/30.0:.10f}'
            out['CI_BURN_INTERVAL_MONTHS']=f'{interval_base:.10f}'
            out['BURN_DELTA_INTERVAL_MONTHS']=f'{burn-interval_base:.10f}'
            out['BURN_DELTA_INTERVAL_MONTHS_PERCENT']=f'{((burn-interval_base)/interval_base*100) if interval_base else float("nan"):.10f}'
            out['PG_RATIO']=f'{burn/pg:.10f}'
            out['MONTHS_RATIO']=f'{burn/mo:.10f}'
            out['INTERVAL_RATIO']=f'{(burn/interval_base) if interval_base else float("nan"):.10f}'
            enriched.append(out); prev=r['_date']
    fields=list(enriched[0].keys())
    with OUTCSV.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(enriched)
    order=['00_first_payment','01_intraday_or_same_day','02_1_to_3_day_burst','03_4_to_14_day_short_cycle','04_15_to_20_day_transition','05_21_to_45_day_monthlyish','06_46_to_59_day_transition','07_60_plus_day_long_gap']
    count_rows=[]; pg_rows=[]; mo_rows=[]; int_rows=[]; fit_rows=[]
    for reg in order:
        sub=[r for r in enriched if r['GAP_REGIME']==reg]
        gaps=[float(r['GAP_DAYS_SINCE_PRIOR']) for r in sub]
        count_rows.append([reg,fmt_int(len(sub)),pct(len(sub)/len(enriched)),fmt_int(len(set(r['ITEMID'] for r in sub))),fmt(np.median(gaps),4),fmt(np.quantile(gaps,.95),4)])
        pg_rows.append([reg]+summ([float(r['BURN_DELTA_PG_PERCENT']) for r in sub]))
        mo_rows.append([reg]+summ([float(r['BURN_DELTA_MONTHS_PERCENT']) for r in sub]))
        int_rows.append([reg]+summ([float(r['BURN_DELTA_INTERVAL_MONTHS_PERCENT']) for r in sub]))
        vals=np.array([float(r['INTERVAL_RATIO']) for r in sub if math.isfinite(float(r['INTERVAL_RATIO']))])
        pos=vals[vals>0]; zero=np.sum(vals==0); neg=vals[vals<0]
        fits=[]
        for name,dist,k in [('Burr XII',stats.burr12,3),('Log-logistic/Fisk',stats.fisk,2),('Lognormal',stats.lognorm,2),('Weibull',stats.weibull_min,2),('Gamma',stats.gamma,2)]:
            if len(pos)>50:
                try:
                    params=dist.fit(pos,floc=0); ll=float(np.sum(dist.logpdf(pos,*params))); aic=2*k-2*ll; ks=stats.kstest(pos,dist.cdf,args=params).statistic
                    fits.append((aic,ks,name,params))
                except Exception: pass
        if fits:
            best=sorted(fits)[0]
            fit_rows.append([reg,pct(len(neg)/len(vals)),pct(zero/len(vals)),pct(len(pos)/len(vals)),best[2],tuple(round(float(x),6) for x in best[3]),fmt(best[0],2),fmt(best[1],5)])
    dist_headers=['Regime','N','Median','P25','P75','P95','P99','Within +/-25','> +100','< -100']
    nb={'cells':[
        md('# Gap-Regime Burn Delta Distribution\n\nThis notebook preprocesses `ci_payment_details_2.csv` by classifying each row into a posting-gap regime, then profiles burn deltas within each regime.'),
        code("import pandas as pd\nclassified = pd.read_csv('ci_payment_details_2_gap_regimes.csv')\nclassified.head()"),
        md('## Regime Counts\n\n' + table(['Regime','Rows','Share','Items represented','Median gap days','P95 gap days'],count_rows)),
        md('## Payment-Group Delta Percent by Regime\n\n' + table(dist_headers,pg_rows)),
        md('## Original Months Delta Percent by Regime\n\n' + table(dist_headers,mo_rows)),
        md('## Interval-Adjusted Months Delta Percent by Regime\n\n' + table(dist_headers,int_rows)),
        md('## Interval Ratio Positive-Component Fits\n\nEach regime is modeled as a negative-correction component, a zero atom, and a positive continuous component. The table reports the best positive-support fit by AIC for `INTERVAL_RATIO = THIS_BURN / CI_BURN_INTERVAL_MONTHS`.\n\n' + table(['Regime','Negative weight','Zero weight','Positive weight','Best positive distribution','SciPy params','AIC','KS'],fit_rows)),
        md('## Interpretation\n\nThe short-gap regimes dominate the file. They should not be interpreted as normal monthly accrual periods; they are payment-event clusters. `_pg` deltas are best centered for payment-event size. Interval-adjusted monthly deltas are useful for calendar allocation, but become heavily right-tailed in short-gap regimes because the exposure denominator is tiny. Long-gap regimes move the other way: interval baselines get large and actual payments often look low versus elapsed-time accrual.')
    ],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
    OUTNB.write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print('wrote',OUTCSV,OUTNB)
if __name__=='__main__': main()
