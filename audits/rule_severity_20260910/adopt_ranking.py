"""Stage, verify, and publish severity outputs and rules-only ranking.

python audits/rule_severity_20260910/adopt_ranking.py stage \
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
  --out-root outputs/review_rule_severity_v11_20260910
Use verify for completed staged exports; publish retains v10 backups.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import statistics
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path[:0]=[str(REPO/'src'), str(REPO/'scripts')]
from rule_severity import SEVERITY_FIELDS, GATE_NAMES
from tae_rule_config import PRODUCTION_RULE_CONFIG_NAME, PRODUCTION_RULE_CONFIG_SHA256
from tae_rule_io import sha256_file
from input_validity import load_input_validity_registry

spec=importlib.util.spec_from_file_location('publisher', HERE.parent/'continuum_monotonic_tail_20260908/publish_regenerated.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
BASELINE=HERE.parent/'continuum_noise_20260910'


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def keyed(path):
    rows=read(path); result={r['mode_key']:r for r in rows}
    assert len(rows)==len(result)
    return result


def shots():
    return [r['shot'] for r in read(BASELINE/'adopted_shot_summary.csv')]


def sources():
    paths=['scripts/tae_rule_engine.py','scripts/tae_rule_config.py','scripts/tae_rule_io.py',
           'scripts/sort_shot_rules.py','scripts/sort_shot_mixed.py','scripts/sort_shot.py',
           'scripts/make_tae_like_list.py','src/cont_features.py','src/continuum_noise.py',
           'src/rule_severity.py','src/mode_features.py','src/nova_mode_loader.py',
           'src/tae_eae_features.py','src/input_validity.py','configs/known_invalid_inputs.csv',
           'configs/rules/tae_rules_production_v10.yaml','configs/rules/tae_rules_production_v11.yaml',
           'audits/continuum_noise_20260910/adopted_shot_summary.csv',
           'audits/continuum_noise_20260910/current_disagreements.csv']
    return {str(REPO/p):sha256_file(REPO/p) for p in paths}


def run_one(task):
    shot,args=task
    original=read(args.rules_root/shot/'all_modes_rules.csv')[0]
    command=[sys.executable,str(REPO/'scripts/sort_shot_mixed.py'), '--method','rules',
             '--shot_dir',str(Path(original['path']).parents[1]),
             '--out_dir',str(args.out_root/'rules'/shot)]
    with (args.out_root/'logs'/f'{shot}.log').open('w') as log:
        subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Regenerated '+shot,flush=True)


def stage(args):
    (args.out_root/'logs').mkdir(parents=True,exist_ok=True)
    snapshot=dict(source_sha256=sources(),stage_driver_sha256=sha256_file(Path(__file__)),
                  old_trees={s:publisher.tree_digest(args.rules_root/s) for s in shots()},
                  ai_trees={s:publisher.tree_digest(args.ai_root/s) for s in shots()})
    p=args.out_root/'run_inputs.json'
    assert not p.exists(), 'Use verify to reuse staged exports'
    p.write_text(json.dumps(snapshot,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_one,[(s,args) for s in shots()]))
    verify(args)


def verify(args):
    snapshot=json.loads((args.out_root/'run_inputs.json').read_text())
    assert snapshot['source_sha256']==sources()
    summaries=[];selection=[];inventory=[];modes=[];new_trees={};counts=Counter();missing=Counter()
    registry=load_input_validity_registry()
    allowed={'rule_features','duplicate_rank_score','duplicate_rank_source','selected_final',
             'rule_configuration_name','rule_configuration_sha256',*SEVERITY_FIELDS}
    for shot in shots():
        assert publisher.tree_digest(args.rules_root/shot)==snapshot['old_trees'][shot]
        assert publisher.tree_digest(args.ai_root/shot)==snapshot['ai_trees'][shot]
        old=keyed(args.rules_root/shot/'all_modes_rules.csv')
        new=keyed(args.out_root/'rules'/shot/'all_modes_rules.csv')
        for key in sorted(old.keys()^new.keys()):
            row=(new if key in new else old)[key]
            assert row['final_decision']=='INVALID' and registry.diagnostic(shot,int(row['n'])),key
            inventory.append(dict(mode_key=key,change='ADDED_INVALID' if key in new else 'ABSENT_INVALID'))
        for key,row in new.items():
            if key not in old:continue
            prev=old[key]
            assert {k:v for k,v in row.items() if k not in allowed}=={k:v for k,v in prev.items() if k not in allowed},key
            if row['processing_status']!='RULE_EVALUATED':continue
            f=json.loads(row['rule_features']);prior=json.loads(prev['rule_features'])
            report=f.pop('severity_features')
            f['feature_schema_version']=prior['feature_schema_version']
            assert f==prior,(key,'prior feature changed')
            assert set(report['gates'])==set(GATE_NAMES)
            for gate,record in report['gates'].items():
                q=record['severity']
                if record['enabled'] and q is not None:
                    assert q>=0 and (q>=1-1e-10 or not record['fired']),(key,gate,'fired below threshold',q)
                    assert q<=1+1e-10 or record['fired'],(key,gate,'passed above threshold',q)
                elif record['enabled']:missing[gate]+=1
            if report['complete']:
                q=float(row['overall_rule_severity'])
                assert abs(float(row['rule_margin'])-(1-q))<1e-12
            counts[row['final_decision']]+=1
            modes.append({k:row[k] for k in ('mode_key','input_fingerprint','final_decision','rule_primary_reason',
                                           'overall_rule_severity','rule_margin','nearest_gate','severity_complete')})
            if row['selected_final']!=prev['selected_final']:
                assert row['final_decision']=='GOOD'
                selection.append(dict(mode_key=key,input_fingerprint=row['input_fingerprint'],
                    previous_selected=prev['selected_final'],selected=row['selected_final'],
                    previous_p_rf=prev['duplicate_rank_score'],severity=row['overall_rule_severity'],nearest_gate=row['nearest_gate']))
        summary=read(args.out_root/'rules'/shot/'shot_summary_wide.csv')[0]
        assert summary['rule_configuration_sha256']==PRODUCTION_RULE_CONFIG_SHA256
        assert summary['duplicate_rank_method']=='rule_severity'
        summaries.append({k:summary[k] for k in ('shot','n_total_files','n_rule_evaluated','n_final_good_before_clustering',
            'n_final_good','n_good_removed_as_duplicates','n_severity_complete','n_severity_unavailable','duplicate_processing_status')})
        new_trees[shot]=publisher.tree_digest(args.out_root/'rules'/shot)
    assert sum(counts.values())==4187 and counts['GOOD']==948,counts
    assert sources()==snapshot['source_sha256']
    write(HERE/'selection_changes.csv',selection,list(selection[0]) if selection else ['mode_key'])
    write(HERE/'inventory_changes.csv',inventory,['mode_key','change'])
    write(HERE/'shot_summary.csv',summaries,list(summaries[0]))
    write(args.out_root/'mode_severities.csv',modes,list(modes[0]))
    populations=defaultdict(list)
    for row in modes:populations[(row['mode_key'].split('/')[0],row['final_decision'])].append(row)
    distribution=[]
    for (shot,decision),rows in sorted(populations.items()):
        values=sorted(float(r['overall_rule_severity']) for r in rows if r['overall_rule_severity'])
        distribution.append(dict(shot=shot,decision=decision,n_modes=len(rows),n_complete=len(values),
            minimum=min(values) if values else None,median=statistics.median(values) if values else None,
            maximum=max(values) if values else None,n_margin_0_to_0p1=sum(.9<=x<=1 for x in values),
            nearest_gate_counts_json=json.dumps(dict(Counter(r['nearest_gate'] for r in rows)),sort_keys=True)))
    write(HERE/'severity_distribution.csv',distribution,list(distribution[0]))
    receipt=dict(config=PRODUCTION_RULE_CONFIG_NAME,config_sha256=PRODUCTION_RULE_CONFIG_SHA256,
                 driver_sha256=sha256_file(Path(__file__)),counts=dict(counts),
                 selected_good=sum(int(r['n_final_good']) for r in summaries),
                 selection_changed_rows=len(selection),inventory_changed_rows=len(inventory),
                 unavailable_gate_counts=dict(missing),new_trees=new_trees,**snapshot)
    (HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Verified: '+json.dumps({k:receipt[k] for k in ('counts','selected_good','selection_changed_rows','unavailable_gate_counts')}),flush=True)


def publish(args):
    receipt=json.loads((HERE/'verification.json').read_text())
    assert sources()==receipt['source_sha256']
    assert sha256_file(Path(__file__))==receipt['driver_sha256']
    backup_root=args.rules_root/'before_rule_severity_v11_20260910'
    staging_root=args.rules_root/'.staging_rule_severity_v11_20260910'
    assert not backup_root.exists() and not staging_root.exists()
    for shot in shots():
        assert publisher.tree_digest(args.out_root/'rules'/shot)==receipt['new_trees'][shot]
        assert publisher.tree_digest(args.rules_root/shot)==receipt['old_trees'][shot]
        assert publisher.tree_digest(args.ai_root/shot)==receipt['ai_trees'][shot]
    for shot in shots():
        shutil.copytree(args.out_root/'rules'/shot,staging_root/shot)
        assert publisher.tree_digest(staging_root/shot)==receipt['new_trees'][shot]
    backup_root.mkdir()
    published=[]
    for shot in shots():
        target=args.rules_root/shot;backup=backup_root/shot
        assert publisher.tree_digest(target)==receipt['old_trees'][shot]
        target.rename(backup)
        try:(staging_root/shot).rename(target)
        except Exception:
            backup.rename(target);raise
        assert publisher.tree_digest(target)==receipt['new_trees'][shot]
        assert publisher.tree_digest(backup)==receipt['old_trees'][shot]
        assert publisher.tree_digest(args.ai_root/shot)==receipt['ai_trees'][shot]
        published.append(dict(shot=shot,output=str(target),backup=str(backup)))
    staging_root.rmdir()
    (HERE/'publication.json').write_text(json.dumps(published,indent=2)+'\n')
    print('Published 27 severity-ranked rules outputs; verified v10 backups and unchanged AI trees.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['stage','verify','publish'])
    for name in ('rules-root','ai-root','out-root'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();globals()[args.phase](args)
