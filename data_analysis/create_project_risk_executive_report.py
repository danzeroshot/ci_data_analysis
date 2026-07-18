from __future__ import annotations

import hashlib, math, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline

DATA = Path('custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv')
DICT = Path('project_feature_non_keyword_field_dictionary_2026-06-10.csv')
DELAY_CORR = Path('approved_keyword_feature_spearman_correlations_with_sources_2026-06-15.csv')
DELAY_HEADLINE = Path('project_delay_rf_approved_keywords_headline_comparison_2026-06-11.csv')
DELAY_IMPORT = Path('project_delay_rf_approved_keywords_before_only_binary_top_importances_2026-06-11.csv')
SPEND_BUCKETS = Path('project_cumulative_spend_elapsed_bucket_summary_allcustomers.csv')
SPEND_MODEL = Path('project_cumulative_spend_model_family_assessment_allcustomers.csv')
SPEND_RAW = Path('custpaydetails_project_clustered_cumulative_curves_allcustomers_2026-06-08-1000.csv')
KW_DETAIL = Path('approved_keyword_feature_column_filter_detail_2026-06-11.csv')
CO_TIME_BIN = Path('change_order_time_lifecycle_bin_summary_2026-06-14.csv')
CO_BUDGET_BIN = Path('change_order_budget_lifecycle_bin_summary_2026-06-14.csv')
CO_LIFT = Path('project_delay_50pct_co_lift_vs_baseline_2026-06-14.csv')
CO_SPARSITY = Path('project_delay_50pct_co_feature_sparsity_diagnostics_2026-06-14.csv')

BUDGET_CORR_OUT = Path('project_budget_overrun_approved_feature_correlations_2026-06-15.csv')
BUDGET_MODEL_OUT = Path('project_budget_overrun_rf_approved_keywords_model_results_2026-06-15.csv')
BUDGET_IMPORTANCE_OUT = Path('project_budget_overrun_rf_approved_keywords_top_importances_2026-06-15.csv')
REPORT_PDF = Path('project_risk_model_executive_summary_2026-06-15.pdf')
REPORT_HTML = Path('project_risk_model_executive_summary_2026-06-15.html')

TARGET_BUDGET = 'PERCENTBUDGETOVERRUN'
IDENTITY_FIELDS = {'RECORD_ID','CUSTOMERNAME','PROJECTID','PROJECTNAME','PROJECTCODE','PROJECTDESCRIPTION','PROJECTSTATUS'}
DATE_FIELDS = {'PLANNEDSTARTDATE','PLANNEDENDDATE'}
INK, MUTED, BLUE, TEAL, ORANGE, RED, GREEN = '#172033','#667085','#2563eb','#0f766e','#f97316','#dc2626','#16a34a'

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','font.family':'DejaVu Sans','axes.edgecolor':'#d0d5dd','axes.labelcolor':INK,'xtick.color':'#344054','ytick.color':'#344054','axes.titleweight':'bold','axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})

def split(row):
    key = f"{row.get('CUSTOMERNAME','')}|{row.get('PROJECTID','')}|{row.name}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8],16) % 100 < 80

def wrap(s,w=88):
    return '\n'.join(textwrap.wrap(str(s), width=w, break_long_words=False))

def header(fig,title,subtitle=''):
    fig.text(.06,.955,title,fontsize=20,weight='bold',color=INK,va='top')
    if subtitle: fig.text(.06,.918,subtitle,fontsize=10.5,color=MUTED,va='top')
    fig.lines.append(plt.Line2D([.06,.94],[.895,.895],transform=fig.transFigure,color='#e4e7ec',lw=1))

def bullets(fig,x,y,items,w=92,size=10.5,gap=.047):
    yy=y
    for item in items:
        t=wrap(item,w)
        fig.text(x,yy,u'\u2022 '+t.replace('\n','\n  '),fontsize=size,color=INK,va='top')
        yy -= gap*(t.count('\n')+1)
    return yy

def style(ax):
    ax.grid(True,axis='y',color='#eef2f6',lw=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#d0d5dd')

def short(s,n=35):
    s=str(s)
    return s if len(s)<=n else s[:n-1]+'...'

def table(ax,df,title):
    ax.axis('off'); ax.set_title(title,loc='left',color=INK,pad=8)
    t=ax.table(cellText=df.values,colLabels=df.columns,cellLoc='left',colLoc='left',loc='upper left')
    t.auto_set_font_size(False); t.set_fontsize(6.5); t.scale(1,1.28)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#eaecf0')
        if r==0: cell.set_facecolor('#f2f4f7'); cell.set_text_props(weight='bold',color=INK)
        else: cell.set_facecolor('white' if r%2 else '#fcfcfd')

def feature_class(c):
    if c.startswith('PROJ_KW_'): return 'project keyword'
    if c.startswith('CONTRACT_KW_'): return 'contract keyword'
    if c.startswith('ITEM_KW_'): return 'item keyword'
    return 'numeric setup feature'

def select_features(df):
    d=pd.read_csv(DICT).fillna('')
    begin={str(r.FieldName).upper():str(r.BeginningAvailable) for _,r in d.iterrows()}
    out=[]
    for c in df.columns:
        cu=c.upper()
        if c in ['PERCENTDELAYED',TARGET_BUDGET] or cu.startswith('TARGET') or cu in IDENTITY_FIELDS or cu in DATE_FIELDS: continue
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if '_KW_' in cu:
            ok=True
        else:
            st=begin.get(cu,'')
            ok=bool(st) and not st.startswith('No') and 'identifier' not in st.lower() and 'excluded' not in st.lower()
        if ok:
            s=pd.to_numeric(df[c],errors='coerce')
            if s.notna().sum()>=20 and s.nunique(dropna=True)>1: out.append(c)
    return out

def budget_correlations(df,features):
    target=pd.to_numeric(df[TARGET_BUDGET],errors='coerce')
    rows=[]
    for c in features:
        x=pd.to_numeric(df[c],errors='coerce'); p=pd.concat([x,target],axis=1).dropna()
        rho=np.nan if len(p)<20 or p.iloc[:,0].nunique()<2 or p.iloc[:,1].nunique()<2 else p.iloc[:,0].corr(p.iloc[:,1],method='spearman')
        direction='zero_or_undefined' if pd.isna(rho) or rho==0 else ('positive' if rho>0 else 'negative')
        rows.append({'direction':direction,'feature':c,'spearman_r':rho,'class':feature_class(c),'non_null_pairs':len(p),'feature_non_null_count':int(x.notna().sum()),'feature_missing_rate':float(1-x.notna().mean())})
    out=pd.DataFrame(rows); out['direction_rank']=pd.NA; out['absolute_rank']=pd.NA
    pos=out[out.direction.eq('positive')].sort_values(['spearman_r','feature'],ascending=[False,True]).index
    neg=out[out.direction.eq('negative')].sort_values(['spearman_r','feature'],ascending=[True,True]).index
    out.loc[pos,'direction_rank']=range(1,len(pos)+1); out.loc[neg,'direction_rank']=range(1,len(neg)+1)
    val=out[out.spearman_r.notna()].assign(abs_r=lambda x:x.spearman_r.abs()).sort_values(['abs_r','feature'],ascending=[False,True]).index
    out.loc[val,'absolute_rank']=range(1,len(val)+1)
    order={'positive':0,'negative':1,'zero_or_undefined':2}; out['_o']=out.direction.map(order); out['_r']=out.direction_rank.fillna(10**9).astype(int)
    out=out.sort_values(['_o','_r','feature']).drop(columns=['_o','_r'])
    out.to_csv(BUDGET_CORR_OUT,index=False)
    return out

def fit_budget(df,features):
    m=df[df[TARGET_BUDGET].notna()].copy(); m=m[np.isfinite(m[TARGET_BUDGET])]
    m['is_train']=m.apply(split,axis=1); tr=m[m.is_train]; te=m[~m.is_train]
    Xtr=tr[features].apply(pd.to_numeric,errors='coerce'); Xte=te[features].apply(pd.to_numeric,errors='coerce')
    ytr=tr[TARGET_BUDGET].astype(float); yte=te[TARGET_BUDGET].astype(float)
    ybtr=(ytr>0).astype(int); ybte=(yte>0).astype(int)
    yttr=pd.cut(ytr,[-np.inf,0,25,np.inf],labels=[0,1,2]).astype(int); ytte=pd.cut(yte,[-np.inf,0,25,np.inf],labels=[0,1,2]).astype(int)
    reg=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=250,max_depth=16,min_samples_leaf=5,min_samples_split=10,max_features=.15,random_state=42,n_jobs=-1))
    binm=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=5,min_samples_split=10,max_features=.15,random_state=42,n_jobs=-1,class_weight='balanced_subsample'))
    trim=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=15,min_samples_split=10,max_features=.15,random_state=42,n_jobs=-1,class_weight='balanced_subsample'))
    reg.fit(Xtr,ytr); pr=reg.predict(Xte)
    binm.fit(Xtr,ybtr); pb=binm.predict_proba(Xte)[:,1]; pbc=(pb>=.5).astype(int)
    trim.fit(Xtr,yttr); pt=trim.predict(Xte); pprob=trim.predict_proba(Xte)
    try: ovr=roc_auc_score(ytte,pprob,multi_class='ovr',labels=trim.classes_)
    except Exception: ovr=np.nan
    rows=[
      {'Target':'budget_overrun','FeatureSet':'before_only','Split':'hash_80_20','Task':'regression','TrainRows':len(tr),'TestRows':len(te),'MAE':mean_absolute_error(yte,pr),'RMSE':math.sqrt(mean_squared_error(yte,pr)),'MSE':mean_squared_error(yte,pr),'BiasMeanPredMinusActual':float(np.mean(pr-yte)),'R2':r2_score(yte,pr),'AUC':np.nan,'BalancedAccuracy':np.nan,'F1':np.nan,'MacroF1':np.nan,'Accuracy':np.nan},
      {'Target':'budget_overrun','FeatureSet':'before_only','Split':'hash_80_20','Task':'binary','TrainRows':len(tr),'TestRows':len(te),'MAE':np.nan,'RMSE':np.nan,'MSE':np.nan,'BiasMeanPredMinusActual':np.nan,'R2':np.nan,'AUC':roc_auc_score(ybte,pb),'BalancedAccuracy':balanced_accuracy_score(ybte,pbc),'F1':f1_score(ybte,pbc),'MacroF1':np.nan,'Accuracy':accuracy_score(ybte,pbc)},
      {'Target':'budget_overrun','FeatureSet':'before_only','Split':'hash_80_20','Task':'three_bin','TrainRows':len(tr),'TestRows':len(te),'MAE':np.nan,'RMSE':np.nan,'MSE':np.nan,'BiasMeanPredMinusActual':np.nan,'R2':np.nan,'AUC':ovr,'BalancedAccuracy':balanced_accuracy_score(ytte,pt),'F1':np.nan,'MacroF1':f1_score(ytte,pt,average='macro'),'Accuracy':accuracy_score(ytte,pt)}]
    res=pd.DataFrame(rows); res.to_csv(BUDGET_MODEL_OUT,index=False)
    imp=pd.DataFrame({'Feature':features,'Importance':binm.named_steps['randomforestclassifier'].feature_importances_}).sort_values('Importance',ascending=False)
    imp.head(100).to_csv(BUDGET_IMPORTANCE_OUT,index=False)
    return {'results':res,'yb_test':ybte,'pbin':pb,'importances':imp,'train':len(tr),'test':len(te)}

def topcorr(corr,direction):
    d=corr[corr.direction.eq(direction)].head(20).copy(); d['rank']=d.direction_rank.astype('Int64'); d['feature']=d.feature.map(lambda x:short(x,35)); d['spearman_r']=d.spearman_r.map(lambda x:'' if pd.isna(x) else f'{x:.3f}')
    return d[['rank','feature','spearman_r','class']]

def build_report(df,features,bcorr,bmodel):
    dcorr=pd.read_csv(DELAY_CORR); dhead=pd.read_csv(DELAY_HEADLINE); sb=pd.read_csv(SPEND_BUCKETS); sm=pd.read_csv(SPEND_MODEL); kw=pd.read_csv(KW_DETAIL); ct=pd.read_csv(CO_TIME_BIN); cb=pd.read_csv(CO_BUDGET_BIN); cl=pd.read_csv(CO_LIFT); cs=pd.read_csv(CO_SPARSITY)
    with PdfPages(REPORT_PDF) as pdf:
        fig=plt.figure(figsize=(11,8.5)); fig.patch.set_facecolor('#f8fafc')
        fig.text(.07,.78,'Project Risk Modeling',fontsize=32,weight='bold',color=INK); fig.text(.07,.72,'Executive Summary of Cumulative Spend, Delay Risk, Budget Risk, and Change Orders',fontsize=15,color=MUTED); fig.text(.07,.64,'Prepared from project-level aggregate data across all customers',fontsize=11,color=MUTED)
        bullets(fig,.09,.52,['Project-level aggregate decision making is the right unit for schedule and budget risk.','Cumulative project spend is broadly S-shaped, with shape changing by duration and observation density.','Beginning-available numerical fields and approved keyword features contain meaningful risk-screening signal.','The models are proof-of-concept screens, not final production models.','Early approved change-order data is sparse and does not yet reliably improve the global delay model.'],95,12,.055)
        fig.text(.07,.08,'Generated 2026-06-15',fontsize=9,color=MUTED); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Project Cumulative Spend Shape','Project-level aggregate spend is nonlinear and better described by percentile bands than a single straight line.')
        gs=GridSpec(2,2,figure=fig,left=.07,right=.95,top=.84,bottom=.12,hspace=.38,wspace=.28); ax=fig.add_subplot(gs[:,0]); x=sb.x_mid*100
        ax.fill_between(x,sb.p10*100,sb.p90*100,color='#bfdbfe',alpha=.55,label='10th-90th'); ax.fill_between(x,sb.p25*100,sb.p75*100,color='#60a5fa',alpha=.35,label='25th-75th'); ax.plot(x,sb['median']*100,color=BLUE,lw=2.6,label='Median'); ax.plot([0,100],[0,100],color='#98a2b3',ls='--',label='Linear')
        ax.set_xlabel('Elapsed project life (%)'); ax.set_ylabel('Cumulative spend (%)'); ax.set_title('Empirical Cumulative Spend Bands'); ax.legend(fontsize=8); style(ax)
        ax2=fig.add_subplot(gs[0,1]); mm=sm.sort_values('MAE'); ax2.barh(mm.ModelFamily,mm.MAE,color=[TEAL if i==0 else '#94a3b8' for i in range(len(mm))]); ax2.invert_yaxis(); ax2.set_xlabel('MAE'); ax2.set_title('Model Family Screen'); style(ax2)
        bullets(fig,.57,.39,['The median curve rises slowly at first, steepens through the middle, then flattens near completion.','The linear reference underestimates spend through much of the middle lifecycle.','Empirical/isotonic and S-curve families are reasonable baselines, but sparse projects and duration strata need separate handling.'],50,10.8,.055)
        pdf.savefig(fig); plt.close(fig)

        raw=pd.read_csv(SPEND_RAW,low_memory=False); raw['duration_bucket']=pd.cut(raw.PROJECTMODELEDDAYS,[-np.inf,60,180,365,np.inf],labels=['<60 days','60-180 days','181-365 days','>365 days']); raw['elapsed_bin']=pd.cut(raw.ELAPSEDPCT,np.linspace(0,1,11),include_lowest=True)
        dur=raw.groupby(['duration_bucket','elapsed_bin'],observed=True).agg(x=('ELAPSEDPCT','median'),y=('CUMULATIVEBURNPCT','median')).reset_index()
        fig=plt.figure(figsize=(11,8.5)); header(fig,'Duration Changes The Spend Curve','Short projects often pay in bursts; long projects compress the initial ramp-up into a small part of the total timeline.')
        ax=fig.add_axes([.08,.22,.58,.58]); colors={'<60 days':RED,'60-180 days':ORANGE,'181-365 days':BLUE,'>365 days':TEAL}
        for b,g in dur.groupby('duration_bucket',observed=True): ax.plot(g.x*100,g.y*100,marker='o',lw=2.2,label=str(b),color=colors[str(b)])
        ax.plot([0,100],[0,100],color='#98a2b3',ls='--'); ax.set_xlabel('Elapsed project life (%)'); ax.set_ylabel('Median cumulative spend (%)'); ax.set_title('Median Spend Curve by Project Duration'); ax.legend(fontsize=9); style(ax)
        bullets(fig,.70,.75,['Short projects under 60 days do not show a stable S curve.','Mid-duration projects show the clearest nonlinear S-shaped behavior.','Long projects over 365 days show less visible early ramp because a fixed 2-3 week mobilization/ramp period is compressed into a small percent of the total project life.','This supports using project duration as a key stratification variable.'],34,10.5,.062)
        pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Feature Engineering And Keyword Review','The feature set combines numerical project setup fields with vetted keyword indicators from project, contract, and item text.')
        ax=fig.add_axes([.08,.48,.42,.30]); stages=['Original\nkeyword groups','After frequency +\nmulti-customer filter','After manual\nsemantic review']; vals=[3000,2069,1940]; ax.bar(stages,vals,color=[BLUE,TEAL,GREEN]); ax.set_ylabel('Keyword family groups'); ax.set_title('Keyword Filtering Funnel'); [ax.text(i,v+45,f'{v:,}',ha='center',fontsize=10,weight='bold') for i,v in enumerate(vals)]; style(ax)
        ax2=fig.add_axes([.58,.48,.33,.30]); fam=kw.groupby(['family','retained_in_output']).size().reset_index(name='count'); ret=fam[fam.retained_in_output==True].set_index('family')['count']; drop=fam[fam.retained_in_output==False].set_index('family')['count']; labs=['project','contract','item']; xx=np.arange(3); ax2.bar(xx,[ret.get(l,0) for l in labs],label='retained',color=GREEN); ax2.bar(xx,[drop.get(l,0) for l in labs],bottom=[ret.get(l,0) for l in labs],label='dropped',color='#fca5a5'); ax2.set_xticks(xx,labs); ax2.set_ylabel('Feature columns'); ax2.set_title('Retained vs Dropped Columns'); ax2.legend(fontsize=8); style(ax2)
        bullets(fig,.08,.36,['Numerical features include planned value, duration, contract/item counts, schedule spread, item price distributions, budget linkage, and related transformations.','Keyword features were generated separately from project descriptions, contract name/description text, and contract-item descriptions.','Automated screening required at least 4 projects and more than 1 customer for each keyword family.','Manual review removed suspect generic, place-specific, mixed-family, abbreviation, and process-noise keywords.','Final approved dataset: 5,762 project rows, 75 non-keyword columns, and 4,141 approved keyword feature columns.'],112,10.7,.047)
        pdf.savefig(fig); plt.close(fig)

        for title,corr,sub in [('Feature Correlations With Schedule Delay',dcorr,'Spearman correlations identify monotonic relationships with PercentDelayed.'),('Feature Correlations With Budget Overrun',bcorr,'Budget overrun target: posted project work completed divided by planned project value, minus 100%.')]:
            fig=plt.figure(figsize=(11,8.5)); header(fig,title,sub); gs=GridSpec(1,2,figure=fig,left=.05,right=.96,top=.84,bottom=.08,wspace=.05); table(fig.add_subplot(gs[0,0]),topcorr(corr,'positive'),'Top Positive Correlations'); table(fig.add_subplot(gs[0,1]),topcorr(corr,'negative'),'Top Negative Correlations'); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Proof-of-Concept Delay Model Performance','Regularized random forest models demonstrate that beginning-available project features contain schedule-risk signal.')
        hh=dhead[dhead.FeatureSet.eq('before_only')].copy(); hh['Label']=hh.Split.str.replace('Hash 80/20','Hash holdout').str.replace('Time old 80/new 20','Time holdout'); ax=fig.add_axes([.08,.48,.84,.30]); idx=np.arange(len(hh)); w=.25; ax.bar(idx-w,hh.Binary_AUC,w,label='Binary AUC',color=BLUE); ax.bar(idx,hh.ThreeBin_BalancedAccuracy,w,label='Three-bin balanced accuracy',color=TEAL); ax.bar(idx+w,hh.Regression_R2,w,label='Regression R2',color=ORANGE); ax.set_xticks(idx,hh.Label); ax.set_ylim(0,1.05); ax.set_title('Delay Model Summary Metrics'); ax.legend(fontsize=9); style(ax)
        ax2=fig.add_axes([.08,.12,.38,.24]); ax2.bar(['MAE','RMSE'],[hh.iloc[0].Regression_MAE,hh.iloc[0].Regression_RMSE],color=[BLUE,TEAL]); ax2.set_title('Hash Holdout Regression Error'); ax2.set_ylabel('PercentDelayed points'); style(ax2)
        bullets(fig,.56,.35,['Binary delayed/not-delayed risk screening is the strongest current use case.','Three-bin severity classification is useful but less stable because exact severity depends on later execution conditions.','The beginning-only model performs very similarly to the broader usable model, supporting deployability after field validation.'],45,10.5,.055)
        pdf.savefig(fig); plt.close(fig)

        br=bmodel['results']; breg=br[br.Task.eq('regression')].iloc[0]; bbin=br[br.Task.eq('binary')].iloc[0]; btri=br[br.Task.eq('three_bin')].iloc[0]
        fig=plt.figure(figsize=(11,8.5)); header(fig,'Proof-of-Concept Budget Overrun Model Performance','Budget target uses completed posted work amount over original planned project value.')
        ax=fig.add_axes([.08,.48,.37,.30]); mets=['Binary AUC','Binary bal. acc.','Three-bin bal. acc.','Three-bin OVR AUC']; vals=[bbin.AUC,bbin.BalancedAccuracy,btri.BalancedAccuracy,btri.AUC]; ax.barh(mets,vals,color=[BLUE,TEAL,ORANGE,GREEN]); ax.set_xlim(0,1); ax.set_title('Budget Classification Metrics'); style(ax)
        ax2=fig.add_axes([.56,.48,.34,.30]); fpr,tpr,_=roc_curve(bmodel['yb_test'],bmodel['pbin']); ax2.plot(fpr,tpr,color=BLUE,lw=2.5,label=f'AUC {bbin.AUC:.3f}'); ax2.plot([0,1],[0,1],color='#98a2b3',ls='--'); ax2.set_xlabel('False positive rate'); ax2.set_ylabel('True positive rate'); ax2.set_title('Budget Overrun ROC'); ax2.legend(); style(ax2)
        ax3=fig.add_axes([.08,.13,.33,.22]); ax3.bar(['MAE','RMSE'],[breg.MAE,breg.RMSE],color=[BLUE,TEAL]); ax3.set_ylabel('Budget-overrun percentage points'); ax3.set_title('Regression Error'); style(ax3)
        bullets(fig,.53,.36,[f'Modeled rows: {int(breg.TrainRows):,} train and {int(breg.TestRows):,} test projects with valid posted work and positive planned project value.','Budget-overrun modeling is newly generated for this report and should be treated as an initial proof of concept.','The target definition depends on posted work completed amount being a reliable final actual-cost proxy.'],48,10.4,.055)
        pdf.savefig(fig); plt.close(fig)

        dimp=pd.read_csv(DELAY_IMPORT).head(15); bimp=bmodel['importances'].head(15); fig=plt.figure(figsize=(11,8.5)); header(fig,'Model Feature Drivers','Random forest importance is not causal, but it shows which setup signals the POC models rely on most.'); gs=GridSpec(1,2,figure=fig,left=.22,right=.95,top=.82,bottom=.10,wspace=.45); ax=fig.add_subplot(gs[0,0]); ax.barh([short(x,28) for x in dimp.Feature[::-1]],dimp.Importance[::-1],color=BLUE); ax.set_title('Delay Binary Model'); style(ax); ax=fig.add_subplot(gs[0,1]); ax.barh([short(x,28) for x in bimp.Feature[::-1]],bimp.Importance[::-1],color=TEAL); ax.set_title('Budget Binary Model'); style(ax); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Change Orders: Lifecycle Timing','Change orders are important controls data, but many occur late enough to be better for monitoring than initial prediction.'); ax=fig.add_axes([.07,.50,.40,.28]); tb=ct[ct.TimingBasis.eq('ApprovedOn')]; ax.bar(tb.LifecycleBin,tb.TotalAddedDays,color=ORANGE); ax.tick_params(axis='x',rotation=45); ax.set_title('Time COs: Added Days by Approval Timing'); ax.set_ylabel('Total added days'); style(ax); ax2=fig.add_axes([.55,.50,.39,.28]); bb=cb[cb.TimingBasis.eq('ApprovedOn')]; ax2.bar(bb.LifecycleBin,bb.NetAmountDelta/1e6,color=TEAL); ax2.tick_params(axis='x',rotation=45); ax2.set_title('Budget COs: Net $ Delta by Approval Timing'); ax2.set_ylabel('Net amount delta ($M)'); style(ax2); bullets(fig,.08,.34,['Lifecycle percentages are not clipped: values above 100% mean the change order was created or approved after the original planned end date.','For schedule-risk prediction, late time extensions are especially important because they may document delay after it is already visible.','For budget-risk monitoring, change orders remain valuable, but their timing limits how much they can improve early project-start prediction.'],112,10.8,.05); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Why Early Change Orders Did Not Improve The Delay Model','At 50% of planned project life, early approved CO features are sparse and unstable in the current dataset.'); ax=fig.add_axes([.08,.48,.40,.30]); sp=cs.head(10); ax.barh([short(x,34) for x in sp.Feature[::-1]],sp.NonZeroShare[::-1]*100,color=BLUE); ax.set_xlabel('Projects with nonzero feature (%)'); ax.set_title('Early CO Feature Sparsity'); style(ax); ax2=fig.add_axes([.56,.48,.34,.30]); lift=cl[cl.Split.eq('hash_80_20')]; ax2.barh(lift.FeatureSet,lift.BalancedAccuracyLiftVsBaseline,color=[GREEN if v>0 else RED for v in lift.BalancedAccuracyLiftVsBaseline]); ax2.axvline(0,color=MUTED,lw=1); ax2.set_title('Lift vs Beginning-Only Baseline'); ax2.set_xlabel('Balanced accuracy lift'); style(ax2); bullets(fig,.08,.34,['Only a small minority of projects have approved change orders by the planned midpoint; early time COs are especially sparse.','CO-only models showed weak standalone predictive power for the three-bin delay target.','Regularized forests tend to prefer broader setup features over sparse CO indicators because setup features help many more project splits.','Next step: keep COs as monitoring/risk-escalation features, but do not depend on them as the primary early prediction signal until more labeled history is available.'],112,10.6,.047); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11,8.5)); header(fig,'Interpretation, Limitations, And Recommended Next Steps','The results support a useful risk-screening direction, with validation items before production use.'); bullets(fig,.08,.80,['Best current use: early project risk screening and prioritization, especially binary delayed/not-delayed and budget-overrun/not-overrun flags.','Do not interpret feature correlations or feature importances as causality. They identify useful signals, not guaranteed causes.','Budget-overrun analysis depends on posted work completed amount being a reliable actual-cost proxy. This should be validated with the client.','Several planned-date and budget-linkage fields need business validation to confirm they are consistently available at the beginning of a project.','The cumulative spend curve should be stratified by duration and observation density before becoming a production expected-spend benchmark.','Recommended next step: build a production candidate around a calibrated risk score, validate field availability with users, and monitor performance over time.'],110,12,.06); fig.text(.08,.15,'Appendix data artifacts generated with this report:',fontsize=11,weight='bold',color=INK); fig.text(.08,.115,wrap(f'{BUDGET_CORR_OUT.name}; {BUDGET_MODEL_OUT.name}; {BUDGET_IMPORTANCE_OUT.name}',120),fontsize=9,color=MUTED); pdf.savefig(fig); plt.close(fig)
    REPORT_HTML.write_text(f'<!doctype html><html><body><h1>Project Risk Modeling Executive Summary</h1><p>Formatted PDF: <code>{REPORT_PDF.name}</code></p><ul><li>{BUDGET_CORR_OUT.name}</li><li>{BUDGET_MODEL_OUT.name}</li><li>{BUDGET_IMPORTANCE_OUT.name}</li></ul></body></html>',encoding='utf-8')

def main():
    df=pd.read_csv(DATA,low_memory=False)
    planned=pd.to_numeric(df.PROJECTPLANNEDVALUE,errors='coerce'); actual=pd.to_numeric(df.TARGETVALIDPOSTEDPROJECTWORKCOMPLETEDAMOUNT,errors='coerce')
    df[TARGET_BUDGET]=np.where((planned>0)&actual.notna(),100.0*actual/planned-100.0,np.nan)
    features=select_features(df)
    bcorr=budget_correlations(df,features)
    bmodel=fit_budget(df,features)
    build_report(df,features,bcorr,bmodel)
    print(f'Feature count used: {len(features):,}')
    for p in [BUDGET_CORR_OUT,BUDGET_MODEL_OUT,BUDGET_IMPORTANCE_OUT,REPORT_PDF,REPORT_HTML]: print('Wrote',p)

if __name__=='__main__': main()
