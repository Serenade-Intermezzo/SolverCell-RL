"""Read-only final resource and controller snapshot; execute via SSH stdin."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().decode().split('\x00')
  if any(arg.endswith('/source/run.py') and '/SolverCell-RL/deployments/' in arg for arg in argv):
   workers.append({'pid':int(p.name),'argv':argv})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
states={}
for name in ('intervention_screen_20261104','temperature_screen_20261104',
             'native_revision_20261104','native_margin_20261104'):
 p=Path('/home/li/SolverCell-RL/deployments')/name/'source/experiments'/name/'controller_state.json'
 d=json.loads(p.read_text());states[name]={'status':d['status'],'jobs':{k:v['status'] for k,v in d['jobs'].items()},'source':d['source_sha256']}
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'training_workers':workers,'campaigns':states,'gpu':gpu,'unrelated_original_worker_exists':Path('/proc/1380376').exists()},indent=2))
