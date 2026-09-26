"""Detached finite training/evaluation/media supervisor. All failures leave status."""
import argparse,json,subprocess,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PY=str(ROOT/'.venv/bin/python')
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
logdir=a.output.parent;logdir.mkdir(parents=True,exist_ok=True)
def run(command,logfile):
 with logfile.open('w') as f:
  proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  return proc.wait()
requested=json.loads(a.config.read_text())
trainer='train_physx_brax.py' if requested.get('source_ppo_port') else 'train_physx.py'
rc=run([PY,trainer,'--config',str(a.config),'--output',str(a.output)],a.output.with_suffix('.training.log'))
statusfile=a.output/'status.json';s=json.loads(statusfile.read_text()) if statusfile.exists() else {}
if not s.get('completed'):
 (a.output.parent/(a.output.name+'.pipeline_error.json')).write_text(json.dumps(dict(stage='training',returncode=rc,status=s),indent=2));raise SystemExit(1)
spec=json.loads((a.output/'declaration.json').read_text());seeds=spec['best_selection']['test_seeds'];cases=[f'seed_{seeds[0]}',f'seed_{seeds[4]}']
report_rc=run([PY,'report_physx_training.py','--run',str(a.output)],a.output/'report.log')
media=[]
for name in ['original_test','best_test']:
 rc=run([PY,'render_traces.py','--input',str(a.output/'evaluation'/name),'--cases',*cases,'--duration','8'],a.output/(name+'_render.log'))
 good=rc==0 and all((a.output/'evaluation'/name/c/'render_verification.json').exists() for c in cases)
 media.append(dict(role=name,returncode=rc,verified_files_present=good))
s=json.loads(statusfile.read_text());s.update(stage='complete' if report_rc==0 and all(x['verified_files_present'] for x in media) else 'media_or_report_failed',media_pending=False,media_results=media,report_returncode=report_rc,pipeline_pid=os.getpid());statusfile.write_text(json.dumps(s,indent=2))
