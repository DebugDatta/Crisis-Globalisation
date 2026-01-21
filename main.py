import os,sys,logging,numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns,yfinance as yf
from scipy.optimize import curve_fit
from scipy import stats
from datetime import datetime
from itertools import product

class Config:
    BASE_DIR="crisis_globalization"
    DIRS=["data/raw","data/processed","outputs/tables","outputs/figures"]
    TICKERS={'US':'^GSPC','UK':'^FTSE','DE':'^GDAXI','JP':'^N225','HK':'^HSI','BR':'^BVSP','IN':'^BSESN'}
    GROUPS={'Developed':['UK','DE','JP','HK'],'Emerging':['BR','IN']}
    VIX_TICKER='^VIX'
    START_DATE='2000-01-01'
    END_DATE=datetime.today().strftime('%Y-%m-%d')
    MAIN_WINDOW=60
    MAIN_PERCENTILE=0.75
    WINDOWS=[30,60,90]
    PERCENTILES=[0.70,0.75,0.80]
    METHODS=['pearson','spearman']

logging.basicConfig(filename='execution_log.txt',level=logging.INFO,format='%(asctime)s - %(message)s',filemode='w')
logger=logging.getLogger(__name__)

def setup_dirs():
    for d in Config.DIRS:os.makedirs(os.path.join(Config.BASE_DIR,d),exist_ok=True)

def fetch_data():
    p_path=os.path.join(Config.BASE_DIR,"data/raw/prices.csv")
    v_path=os.path.join(Config.BASE_DIR,"data/raw/vix.csv")
    if os.path.exists(p_path) and os.path.exists(v_path):
        prices=pd.read_csv(p_path,index_col=0,parse_dates=True)
        vix=pd.read_csv(v_path,index_col=0,parse_dates=True)
        if isinstance(vix,pd.DataFrame):vix=vix.iloc[:,0]
    else:
        prices=yf.download(list(Config.TICKERS.values()),start=Config.START_DATE,end=Config.END_DATE,auto_adjust=True,ignore_tz=True)['Close']
        if isinstance(prices.columns,pd.MultiIndex):prices.columns=prices.columns.get_level_values(0)
        inv_map={v:k for k,v in Config.TICKERS.items()}
        prices.rename(columns=inv_map,inplace=True)
        vix_data=yf.download(Config.VIX_TICKER,start=Config.START_DATE,end=Config.END_DATE,auto_adjust=True,ignore_tz=True)
        vix=vix_data['Close'] if 'Close' in vix_data.columns else vix_data
        if isinstance(vix,pd.DataFrame):vix=vix.iloc[:,0]
        vix.name='VIX'
        prices.to_csv(p_path);vix.to_csv(v_path)
    prices=prices.ffill().dropna();vix=vix.ffill().dropna()
    common=prices.index.intersection(vix.index)
    return prices.loc[common],vix.loc[common]

def get_returns(prices):
    return np.log(prices/prices.shift(1)).dropna()

def compute_rolling_metrics(returns,window):
    df=pd.DataFrame(index=returns.index)
    target=returns['US']
    for col in returns.columns:
        if col!='US':df[col]=target.rolling(window).corr(returns[col])
    df['Global_Avg']=df.mean(axis=1)
    df['Developed_Avg']=df[Config.GROUPS['Developed']].mean(axis=1)
    df['Emerging_Avg']=df[Config.GROUPS['Emerging']].mean(axis=1)
    return df

def define_regimes(vix,percentile):
    q=vix.quantile(percentile)
    thresh=q.item() if hasattr(q,"item") else float(q)
    return (vix>thresh).astype(int),thresh

def test_regime_switch_robustness(returns,vix):
    results=[]
    for w,p,m in product(Config.WINDOWS,Config.PERCENTILES,Config.METHODS):
        c_df=compute_rolling_metrics(returns,w)
        metric=c_df['Global_Avg']
        flags,_=define_regimes(vix,p)
        flags=flags.reindex(metric.index).fillna(0)
        crisis_vals=metric[flags==1].dropna()
        stable_vals=metric[flags==0].dropna()
        if len(crisis_vals)<2 or len(stable_vals)<2:continue
        t_stat,p_val=stats.ttest_ind(crisis_vals,stable_vals,equal_var=False)
        c_mean,s_mean=crisis_vals.mean(),stable_vals.mean()
        results.append({'Window':w,'Percentile':p,'Method':m,'Crisis_Corr':c_mean,'Stable_Corr':s_mean,'Delta':c_mean-s_mean,'T_Stat':t_stat,'P_Value':p_val,'Significant_95':p_val<0.05})
    return pd.DataFrame(results)

def test_structural_relationship(vix,corr_series):
    common=corr_series.index.intersection(vix.index)
    x,y=vix.loc[common],corr_series.loc[common]
    mask=~np.isnan(x)&~np.isnan(y)
    x,y=x[mask],y[mask]
    slope,intercept,r_val,p_val,stderr=stats.linregress(x,y)
    return {'slope':slope,'intercept':intercept,'r2':r_val**2,'p_val':p_val,'stderr':stderr}

def model_decay_dynamics(corr_series,flags):
    flags=flags.reindex(corr_series.index).fillna(0)
    starts=flags.diff()[flags.diff()==1].index
    curves=[]
    for s in starts:
        try:
            loc=corr_series.index.get_loc(s)
            if loc+60<len(corr_series):curves.append(corr_series.iloc[loc:loc+60].values)
        except:pass
    if not curves:return None,None
    avg_curve=np.nanmean(curves,axis=0)
    def model(t,a,l,b):return a*np.exp(-l*t)+b
    try:
        popt,_=curve_fit(model,np.arange(60),avg_curve,p0=[0.5,0.1,0.3],maxfev=5000)
        return popt,avg_curve
    except:return None,avg_curve

def save_plot(filename,dpi=300):
    plt.savefig(os.path.join(Config.BASE_DIR,"outputs/figures",filename),dpi=dpi,bbox_inches='tight');plt.close()

def generate_report_visuals(prices,returns,vix,corr_main,flags_main,thresh_main,ols_res,decay_res,robust_df):
    sns.set_theme(style="whitegrid")
    flags_main=flags_main.loc[corr_main.index]
    plt.figure(figsize=(12,6))
    norm=prices/prices.iloc[0]*100
    for c in prices.columns:
        lw=2.5 if c=='US' else 1;alpha=1.0 if c=='US' else 0.6
        plt.plot(norm.index,norm[c],label=c,linewidth=lw,alpha=alpha)
    plt.title("Figure 1: Global Market Trajectories");plt.ylabel("Normalized Price");plt.legend(bbox_to_anchor=(1.01,1),loc='upper left')
    save_plot("01_market_overview.png")
    plt.figure(figsize=(12,5))
    x=np.arange(len(vix));y=vix.values.astype(float);mask=y>float(thresh_main)
    plt.plot(x,y,color='#333333',lw=1,label='VIX')
    plt.axhline(float(thresh_main),color='red',ls='--',label=f'Threshold ({int(Config.MAIN_PERCENTILE*100)}th %)')
    plt.fill_between(x,np.zeros_like(y),y,where=mask,alpha=0.3,label='Crisis')
    plt.title("Figure 2: Crisis Regimes via VIX");plt.legend()
    save_plot("02_crisis_identification.png")
    c_calm=returns[flags_main==0].corr();c_crisis=returns[flags_main==1].corr()
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(c_calm,annot=True,cmap="coolwarm",vmin=0,vmax=1,ax=axes[0]);axes[0].set_title("STABLE")
    sns.heatmap(c_crisis,annot=True,cmap="coolwarm",vmin=0,vmax=1,ax=axes[1]);axes[1].set_title("CRISIS")
    plt.suptitle("Figure 3: Correlation Structure Shift");save_plot("03_correlation_matrices.png")
    plt.figure(figsize=(12,6))
    plt.plot(corr_main.index,corr_main['Global_Avg'],color='navy',lw=1)
    plt.fill_between(corr_main.index,0,1,where=(flags_main==1),color='red',alpha=0.15,label='Crisis')
    plt.title("Figure 4: Global Integration Index");plt.ylim(0,1);save_plot("04_rolling_integration.png")
    plt.figure(figsize=(12,6))
    plt.plot(corr_main.index,corr_main['Developed_Avg'],label='Developed',color='blue',alpha=0.7)
    plt.plot(corr_main.index,corr_main['Emerging_Avg'],label='Emerging',color='orange',alpha=0.8)
    plt.title("Figure 5: Developed vs Emerging");plt.legend();save_plot("05_regional_comparison.png")
    popt,curve=decay_res
    if curve is not None:
        plt.figure(figsize=(8,5));t=np.arange(len(curve))
        plt.scatter(t,curve,color='black',s=10,label='Observed')
        if popt is not None:plt.plot(t,popt[0]*np.exp(-popt[1]*t)+popt[2],'r-',lw=3,label='Model')
        plt.title("Figure 6: Decay Dynamics");plt.legend();save_plot("06_decay_dynamics.png")
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Window',y='Delta',hue='Percentile',data=robust_df)
    plt.title("Figure 7: Robustness Distribution");plt.ylabel("Delta");save_plot("07_robustness_check.png")
    c_idx=corr_main.index.intersection(vix.index)
    vx,cy=vix.loc[c_idx],corr_main.loc[c_idx,'Global_Avg']
    plt.figure(figsize=(10,6))
    plt.scatter(vx,cy,alpha=0.1,color='gray',s=10)
    plt.plot(vx,ols_res['slope']*vx+ols_res['intercept'],color='red',lw=2,label=f"R2={ols_res['r2']:.2f}")
    plt.title("Figure 8: OLS Regression");plt.xlabel("VIX");plt.ylabel("Correlation");plt.legend()
    save_plot("08_regression_analysis.png")

def main():
    setup_dirs()
    prices,vix=fetch_data()
    returns=get_returns(prices)
    corr_main=compute_rolling_metrics(returns,Config.MAIN_WINDOW)
    flags_main,thresh_main=define_regimes(vix,Config.MAIN_PERCENTILE)
    robust_df=test_regime_switch_robustness(returns,vix)
    ols_results=test_structural_relationship(vix,corr_main['Global_Avg'])
    decay_results=model_decay_dynamics(corr_main['Global_Avg'],flags_main)
    desc_stats=returns.describe().transpose()
    robust_df.to_csv(os.path.join(Config.BASE_DIR,"outputs/tables/robustness_full.csv"),index=False)
    desc_stats.to_csv(os.path.join(Config.BASE_DIR,"outputs/tables/descriptive_stats.csv"))
    with open(os.path.join(Config.BASE_DIR,"outputs/tables/regression_summary.txt"),"w") as f:
        f.write(f"Slope: {ols_results['slope']:.5f}\nR2: {ols_results['r2']:.5f}\nP-Val: {ols_results['p_val']:.5e}\n")
    generate_report_visuals(prices,returns,vix,corr_main,flags_main,thresh_main,ols_results,decay_results,robust_df)
    logger.info("Pipeline Completed.")

if __name__=="__main__":main()
