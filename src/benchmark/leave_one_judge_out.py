"""Offline judge sensitivity using recorded model identities, never slot aliases.

The frozen headline decisions remain unchanged. Later verdicts and normalizations
are a separate supplementary snapshot. Missing canonical judges remain missing.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CANONICAL=('openai:gpt-5.5','anthropic:claude-opus-4-7','google:gemini-3.5-flash','openrouter:deepseek/deepseek-v4-pro')
FAMILIES={'gpt':'openai','gpt-5.6-sol':'openai','gpt-5.6-luna':'openai','gpt-5.6-terra':'openai','opus':'anthropic','opus-4-8':'anthropic','sonnet':'anthropic','gemini':'google','gemini-3.1':'google','deepseek':'deepseek','deepseek-v4':'deepseek','deepseek-v4-pro':'deepseek'}
def identity(record):
    if not isinstance(record,dict):return None
    provider=record.get('provider');name=record.get('model_name')
    if provider and name:return f'{provider}:{name}'
    model=record.get('model') or record.get('label')
    return model if isinstance(model,str) and ':' in model else None

def vote(record):
    if not isinstance(record,dict) or record.get('error'):return None
    answer=record.get('answer') or {};v=answer.get('verdict')
    return v if v in ('yes','no') else None

def family(model):
    if 'deepseek' in model:return 'deepseek'
    return model.split(':',1)[0]

def select_votes(row):
    selected={};sources={}
    for slot,record in row['panel'].items():
        if slot not in ('judge_1','judge_2','judge_3'):continue
        name=identity(record);v=vote(record)
        if name and v:selected[name]=v;sources[name]='panel/'+slot
    for stem,payload in sorted(row['per_judge'].items()):
        record=payload['record'];name=identity(record) or identity(payload['judge']);v=vote(record)
        if name and v:selected[name]=v;sources[name]='per_judge/'+stem
    return selected,sources

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'analysis/generated/judges');args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=json.loads((ROOT/'data/analysis/judges/records.json').read_text());headline=json.loads((ROOT/'data/results/paper_results.json').read_text());head={(m,s,r['id']):r for m,ms in headline.items() for s,rs in ms.items() for r in rs}
    groups=defaultdict(list);identities=Counter();ledger=[];checker_drift=[];normalization_drift=[];self_rates=defaultdict(lambda:[0,0]);panel_checks=[]
    for row in raw:
        key=(row['configuration'],row['mode'],row['id']);h=head[key]
        if h['correct']!=row['checker']['correct']:checker_drift.append(list(key))
        if any(h.get(k)!=row['checker'].get(k) for k in ['normalized_prediction','normalized_reference']):normalization_drift.append(list(key))
        vs,sources=select_votes(row);groups[key[:2]].append((row,vs));identities.update(vs.keys())
        for name,v in vs.items():
            ledger.append({'configuration':key[0],'mode':key[1],'id':key[2],'judge_model':name,'verdict':v,'source':sources[name]})
            if not row['checker']['correct']:
                own=family(name)==FAMILIES.get(key[0],key[0]);cell=self_rates[(name,'own_family' if own else 'other_family')];cell[0]+=v=='yes';cell[1]+=1
    # All frozen panel votes are checked as stored, independently of supplementary votes.
    for key,h in head.items():
        votes=[vote(h.get(slot)) for slot in ('judge_1','judge_2','judge_3')];valid=[v for v in votes if v];reconstructed=bool(h['correct']) or (valid.count('yes')>valid.count('no'))
        if reconstructed!=h['final_vote']:panel_checks.append(list(key))
    results=[]
    for (m,mode),entries in groups.items():
        for triple in combinations(CANONICAL,3):
            accepted=unknown=0
            for row,vs in entries:
                if row['checker']['correct']:accepted+=1;continue
                vals=[vs.get(j) for j in triple]
                if None in vals:unknown+=1;continue
                accepted+=vals.count('yes')>=2
            results.append({'configuration':m,'mode':mode,'dropped_real_judge':next(j for j in CANONICAL if j not in triple),'accepted_lower':accepted,'unavailable_three_votes':unknown,'accuracy_lower':accepted/133,'accuracy_upper':(accepted+unknown)/133})
        accepted=missing=0
        for row,vs in entries:
            if row['checker']['correct']:accepted+=1;continue
            vals=[v for name,v in vs.items() if family(name)!=FAMILIES.get(m,m)]
            if not vals:missing+=1;continue
            accepted+=vals.count('yes')>vals.count('no')
        results.append({'configuration':m,'mode':mode,'dropped_real_judge':'all recorded votes from generator family','accepted_lower':accepted,'unavailable_three_votes':missing,'accuracy_lower':accepted/133,'accuracy_upper':(accepted+missing)/133})
    for name,rows in [('sensitivity.csv',results),('selected_votes.csv',ledger),('family_yes_rates.csv',[{'judge_model':k[0],'generator_group':k[1],'yes':v[0],'valid':v[1],'yes_rate':v[0]/v[1]} for k,v in sorted(self_rates.items())])]:
        with (args.output_dir/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit={'real_selected_judge_models':dict(identities),'supplementary_checker_drift_from_headline':checker_drift,'supplementary_normalization_drift_from_headline':normalization_drift,'frozen_panel_reconstruction_mismatches':panel_checks,'canonical_triple_policy':'require all three valid real-model votes; otherwise publish lower/upper bounds; no substitute-as-canonical fallback','family_policy':'exclude every selected actual model in generator family; use majority of remaining valid votes','selection_policy':'later per-model file preferred over panel record of same real identity; both raw records retained','legacy_warning':'legacy_slot_analysis.json relabelled substitute votes by panel slot; retained as historical analysis, not a canonical-model exclusion result'}
    (args.output_dir/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['supplementary_checker_drift_from_headline','supplementary_normalization_drift_from_headline']},indent=2));print('checker drift',len(checker_drift),'normalization drift',len(normalization_drift))
if __name__=='__main__':main()
