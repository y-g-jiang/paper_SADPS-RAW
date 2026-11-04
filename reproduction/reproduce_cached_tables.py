"""Portable, standard-library-only verification of the bundled frozen metrics."""
import argparse, collections, csv, json, math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.bundle; groups=collections.defaultdict(list);keys=['KL','J_CDF','NLL','CRPS','mean_MSE','coverage90','zero_atom_error']
    with (root/'independent_test/per_case_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        if int(r['n']):groups[r['method'],r['scene']].append(r)
    summary=json.loads((root/'independent_test/bootstrap_summary.json').read_text(encoding='utf-8-sig'))['summaries']
    maxerr=0.;checks=0
    for method,expected in summary.items():
        mothers=[rr for (m,s),rr in sorted(groups.items()) if m==method]
        assert len(mothers)==expected['mothers']==138
        for key in keys:
            value=sum(sum(float(r[key]) for r in rr)/len(rr) for rr in mothers)/len(mothers)
            err=abs(value-expected['metrics'][key]['mean']);maxerr=max(maxerr,err)
            assert math.isclose(value,expected['metrics'][key]['mean'],rel_tol=1e-11,abs_tol=1e-11),(method,key,value)
            checks+=1
    bycase=collections.defaultdict(list)
    for r in rows:bycase[r['scene'],r['realization'],r['context']].append(r)
    assert len(bycase)==552
    for rr in bycase.values():
        assert len(rr)==6 and len({r['input_sha256'] for r in rr})==1 and len({r['n'] for r in rr})==1
    result=dict(status='PASS',rows=len(rows),cases=len(bycase),methods=len(summary),mothers=138,mean_checks=checks,max_absolute_mean_error=maxerr,
        scope='Reproduces cached six-method macro means and paired-input identities, not neural inference or a rerun of the independent test')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
