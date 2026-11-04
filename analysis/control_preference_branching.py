"""One CPU-bounded, durable controller for a fixed branching campaign stage."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--checkpoint-root',required=True,type=Path)
    p.add_argument('--python',required=True)
    p.add_argument('--selection',type=Path)
    p.add_argument('--stage',choices=('screen','final'),required=True)
    args=p.parse_args();args.runtime.mkdir(parents=True,exist_ok=True)
    lock=(args.runtime/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(args.manifest.read_text());campaign=manifest['campaign_id']
    target=args.runtime/f'{args.stage}_controller.json'
    if target.exists():raise FileExistsError(target)
    jobs=[{'seed':seed,'status':'queued'} for seed in ([0] if args.stage=='screen' else [0,1,2])]
    active={};files=[]
    def save():
        value={'stage':args.stage,'pid':os.getpid(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'status':'running' if active or any(j['status']=='queued' for j in jobs) else 'complete', 'jobs':jobs}
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(target)
    while any(j['status'] in ('queued','running') for j in jobs):
        for cpu,process in list(active.items()):
            rc=process.poll()
            if rc is not None:
                job=next(j for j in jobs if j.get('pid')==process.pid)
                job.update(status='complete' if rc==0 else 'failed',returncode=rc,finished=time.time());del active[cpu];save()
        for cpu in (24,25):
            if args.stage=='screen' and cpu==25:continue
            if cpu in active:continue
            job=next((j for j in jobs if j['status']=='queued'),None)
            if job is None:continue
            name='screen' if args.stage=='screen' else f"final_seed{job['seed']}"
            output=Path('outputs')/campaign/name
            if output.exists():raise FileExistsError(output)
            cmd=[args.python,'analysis/run_preference_branching.py','--manifest',str(args.manifest),'--output',str(output),
                 '--checkpoint-root',str(args.checkpoint_root),'--stage',args.stage,'--seed',str(job['seed'])]
            if args.selection:cmd+=['--selection',str(args.selection)]
            stream=(args.runtime/f'{name}.log').open('w');files.append(stream)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            process=subprocess.Popen(['taskset','-c',str(cpu)]+cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(status='running',pid=process.pid,cpu=cpu,started=time.time(),command=cmd,output=str(output));active[cpu]=process;save()
        if active:time.sleep(2)
    save()
    for stream in files:stream.close()
    raise SystemExit(0 if all(j['status']=='complete' for j in jobs) else 1)


if __name__=='__main__':main()
