import json,glob,numpy as np,os
D=os.path.dirname(os.path.abspath(__file__))
R=[json.load(open(f)) for f in glob.glob(f'{D}/out/*.json')]
print(len(R))
for sel,name in ((lambda r:True,'all'),(lambda r:r['n1']>=60,'n1>=60'),(lambda r:r['n1']<=40,'n1<=40')):
    S=[r for r in R if sel(r)]
    cen=np.array([max(r['lb_u1'],r['lb_rev']) for r in S]); lp=np.array([max(r['f_lp'],r['r_lp']) for r in S]); li=np.array([max(r['f_lpint'],r['r_lpint']) for r in S])
    lo=np.array([np.ceil(r['lo']-1e-9) for r in S]); lpraw=np.array([max(r['f_lp_raw'],r['r_lp_raw']) for r in S]); liraw=np.array([max(r['f_lpint_raw'],r['r_lpint_raw']) for r in S])
    nV=np.array([max(r['f_nV'],r['r_nV']) for r in S]); cpu=np.array([r['f_cpu_lp']+r['r_cpu_lp'] for r in S])
    l0=np.array([r['lo'] for r in S])
    print(name,len(S),'| LP>census',int((lp>cen).sum()),'| LP raw > L0:',np.mean(lpraw>l0+1e-9),'| LPint raw > L0:',np.mean(liraw>l0+1e-9),
      '| mean (census-LP)/census %.2f%%'%(100*np.mean((cen-lp)/cen)),'| mean (census-L0)/census %.2f%%'%(100*np.mean((cen-lo)/cen)),
      '| mean LPraw/L0 %.4f'%np.mean(lpraw/l0),'| nV median/max',np.median(nV),nV.max(),'| cpu fwd+rev mean/max %.3f/%.3f s'%(cpu.mean(),cpu.max()))
