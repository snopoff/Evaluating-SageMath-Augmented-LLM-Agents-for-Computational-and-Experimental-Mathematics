"""Offline analysis of the frozen manuscript cohort. No provider or judge calls."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np

# Keep generated caches within a writable temporary directory on restricted hosts.
import os, tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'anonymous-release-matplotlib'))

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / 'data/results/paper_results.json'
MODEL_LABELS = json.loads((ROOT / 'data/results/model_labels.json').read_text())
SEED = 20260727

def to_bool(value):
    if isinstance(value, bool): return value
    if value in ('true', 'True', 'yes', 1): return True
    if value in ('false', 'False', 'no', 0): return False
    raise ValueError(f'Invalid stored boolean: {type(value).__name__}')

def exact_mcnemar_p(direct_only, tool_only):
    n = direct_only + tool_only
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(direct_only,tool_only)+1)) / 2**n) if n else 1.0

def bootstrap(values, rng, samples=20000):
    values=np.asarray(values,dtype=float); draws=[]
    for start in range(0,samples,2000):
        idx=rng.integers(0,len(values),(min(2000,samples-start),len(values)))
        draws.extend(values[idx].mean(axis=1))
    return tuple(float(x) for x in np.quantile(draws,[.025,.975]))

def holm(values):
    out=[0.]*len(values); maximum=0.
    for rank,i in enumerate(sorted(range(len(values)),key=values.__getitem__)):
        maximum=max(maximum,(len(values)-rank)*values[i]);out[i]=min(1.,maximum)
    return out

def model_rows(data, samples=200000):
    ids=[r['id'] for r in json.loads((ROOT/'data/processed/problems_to_run.json').read_text())]
    if len(ids)!=133 or len(set(ids))!=133 or set(data)!=set(MODEL_LABELS):raise ValueError('Unexpected cohort')
    rows=[];rng=np.random.default_rng(SEED)
    for model,label in MODEL_LABELS.items():
        modes=data[model]
        if set(modes)!={'tool','no_tool'}:raise ValueError('Unexpected modes')
        cells={mode:{r['id']:r for r in rs} for mode,rs in modes.items()}
        for mode, rs in modes.items():
            if len(rs)!=133 or set(cells[mode])!=set(ids):raise ValueError('Unaligned IDs')
        direct=np.array([to_bool(cells['no_tool'][i]['final_vote']) for i in ids])
        tool=np.array([to_bool(cells['tool'][i]['final_vote']) for i in ids])
        dif=tool.astype(float)-direct.astype(float);lo,hi=bootstrap(dif,rng,samples)
        a=int((direct & ~tool).sum());b=int((tool & ~direct).sum())
        rows.append(dict(model_id=model,model=label,n=len(ids),direct_correct=int(direct.sum()),tool_correct=int(tool.sum()),direct_rate=float(direct.mean()),tool_rate=float(tool.mean()),gain=float(dif.mean()),gain_ci_lo=lo,gain_ci_hi=hi,direct_only=a,tool_only=b,mcnemar_p=exact_mcnemar_p(a,b)))
    return rows,ids

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def cluster_ci(values, clusters, rng, samples=20000):
    keys=sorted(set(clusters));sums=np.array([sum(v for v,c in zip(values,clusters) if c==key) for key in keys]);counts=np.array([clusters.count(key) for key in keys]);draws=[]
    for start in range(0,samples,2000):
        idx=rng.integers(0,len(keys),(min(2000,samples-start),len(keys)))
        draws.extend(sums[idx].sum(axis=1)/counts[idx].sum(axis=1))
    return [float(v) for v in np.quantile(draws,[.025,.975])]

def analyze(output_dir, samples=20000):
    data=json.loads(RESULTS_PATH.read_text());rows,ids=model_rows(data,samples);output_dir.mkdir(parents=True,exist_ok=True)
    by={m:{s:{r['id']:r for r in rs} for s,rs in modes.items()} for m,modes in data.items()}
    matrix={s:np.array([[to_bool(by[m][s][i]['final_vote']) for i in ids] for m in MODEL_LABELS],dtype=float) for s in ('no_tool','tool')}
    symbolic={s:np.array([[to_bool(by[m][s][i]['correct']) for i in ids] for m in MODEL_LABELS],dtype=float) for s in matrix}
    diff=matrix['tool']-matrix['no_tool'];sdiff=symbolic['tool']-symbolic['no_tool'];rng=np.random.default_rng(SEED)
    ref=by['gpt']['no_tool'];clusters=[ref[i]['arxiv_id'] for i in ids]
    adjusted=holm([r['mcnemar_p'] for r in rows])
    for r,p in zip(rows,adjusted):r['mcnemar_p_holm_20']=p
    summary={'configurations':20,'items':133,'evaluations':5320,'source_papers':len(set(clusters)),'hybrid_accepts':{s:int(v.sum()) for s,v in matrix.items()},'symbolic_accepts':{s:int(v.sum()) for s,v in symbolic.items()},'mean_hybrid_gain_pp':100*float(diff.mean()),'mean_symbolic_gain_pp':100*float(sdiff.mean()),'hybrid_item_bootstrap_ci_pp':[100*v for v in bootstrap(diff.mean(axis=0),rng,samples)],'hybrid_source_paper_bootstrap_ci_pp':[100*v for v in cluster_ci(diff.mean(axis=0),clusters,rng,samples)],'symbolic_source_paper_bootstrap_ci_pp':[100*v for v in cluster_ci(sdiff.mean(axis=0),clusters,rng,samples)],'holm_significant_models':[r['model'] for r in rows if r['mcnemar_p_holm_20']<.05],'bootstrap_samples':samples,'seed':SEED,'variance_scope':'items/source papers; no repeated-generation variance'}
    write_csv(output_dir/'paired_model_statistics.csv',rows)
    types=[];categories=[]
    for field,target in [('answer_type',types),('arxiv_tag',categories)]:
        for category in sorted({ref[i][field] for i in ids}):
            indexes=[j for j,i in enumerate(ids) if ref[i][field]==category]
            target.append({'group':category,'n':len(indexes),'direct_rate':float(matrix['no_tool'][:,indexes].mean()),'tool_rate':float(matrix['tool'][:,indexes].mean()),'gain_pp':100*float(diff[:,indexes].mean())})
    write_csv(output_dir/'answer_type.csv',types);write_csv(output_dir/'category_accuracy.csv',categories)
    traces=[];failures=Counter()
    for m in MODEL_LABELS:
        rs=data[m]['tool'];calls=[t for r in rs for t in r.get('tool_traces',[])];names=Counter(t.get('tool_name') or t.get('name') for t in calls)
        for t in calls:
            if (t.get('tool_name') or t.get('name'))=='sage_exec' and not t.get('ok'):
                meta=t.get('metadata') or {};text=str(meta.get('traceback',''))+' '+str(meta.get('error',''))+' '+str(t.get('content',''))
                import re
                found=re.findall(r'\b([A-Za-z]+(?:Error|Exception))\b',text)
                failures[(m,found[-1] if found else str(meta.get('error_kind') or 'unclassified'))]+=1
        traces.append({'model':MODEL_LABELS[m],'tracked_tool_calls':len(calls),'sage_calls':names['sage_exec'],'context7_calls':sum(n for k,n in names.items() if k in ['query-docs','resolve-library-id']),'mean_turn_count':float(np.mean([r.get('turn_count',0) for r in rs])),'mean_tracked_calls':len(calls)/133,'budget_turns_ge_15':sum(r.get('turn_count',0)>=15 for r in rs)})
    write_csv(output_dir/'tool_usage.csv',traces)
    write_csv(output_dir/'sage_failures.csv',[{'model':MODEL_LABELS[m],'exception':e,'count':n} for (m,e),n in sorted(failures.items())])
    # Scientific figures use stored rows only; no new prices or model calls.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[r['model'] for r in rows];y=np.arange(len(rows))
    fig,axs=plt.subplots(1,3,figsize=(16,9),sharey=True)
    axs[0].barh(y-.18,[100*r['direct_rate'] for r in rows],height=.35,label='No tool');axs[0].barh(y+.18,[100*r['tool_rate'] for r in rows],height=.35,label='Sage workflow');axs[0].set_yticks(y,names);axs[0].legend();axs[0].set_xlabel('Hybrid solve rate (%)')
    axs[1].barh(y,[100*r['gain'] for r in rows]);axs[1].errorbar([100*r['gain'] for r in rows],y,xerr=[[100*(r['gain']-r['gain_ci_lo']) for r in rows],[100*(r['gain_ci_hi']-r['gain']) for r in rows]],fmt='none',color='black');axs[1].set_xlabel('Paired gain (pp), item bootstrap 95% CI')
    tokens=[]
    for m in MODEL_LABELS:
        solved=[r for r in data[m]['tool'] if to_bool(r['final_vote'])]
        tokens.append(np.mean([r.get('token_usage',{}).get('total_tokens',0) for r in solved]))
    axs[2].barh(y,tokens);axs[2].set_xscale('log');axs[2].set_xlabel('Mean reported tokens / successful run');axs[0].invert_yaxis();fig.tight_layout();fig.savefig(output_dir/'accuracy_gain_tokens.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.hist([len(r.get('tool_traces',[])) for m in data for r in data[m]['tool']],bins=np.arange(0,19)-.5);ax.set_xlabel('Tracked executed tool calls per run');ax.set_ylabel('Runs');fig.tight_layout();fig.savefig(output_dir/'tool_calls.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,s in zip(axs,matrix):
        agreement=(matrix[s][:,None,:]==matrix[s][None,:,:]).mean(axis=2);im=ax.imshow(agreement,vmin=.4,vmax=1);ax.set_title(s);ax.set_xticks(y,names,rotation=90,fontsize=6);ax.set_yticks(y,names,fontsize=6)
    fig.colorbar(im,ax=axs.ravel().tolist());fig.savefig(output_dir/'agreement.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'analysis/generated');parser.add_argument('--samples',type=int,default=20000);args=parser.parse_args();analyze(args.output_dir,args.samples)
if __name__=='__main__':main()
