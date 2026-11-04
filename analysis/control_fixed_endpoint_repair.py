"""Single durable CPU24/25 controller for the frozen fixed-endpoint campaign."""
import argparse,fcntl,hashlib,json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json,source_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--runtime',required=True,type=Path);args=p.parse_args();args.runtime.mkdir(parents=True,exist_ok=True)
    lock=(args.runtime/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=args.runtime/'controller.json'
    if target.exists():raise FileExistsError('Fresh controller only')
    m=json.loads(args.manifest.read_text());assert source_manifest()['source_sha256']==m['source_sha256']
    assert {24,25}<=os.sched_getaffinity(0)
    jobs=[dict(j,status='queued') for j in m['jobs']];active={};handles=[]
    def save():atomic_json(target,{'pid':os.getpid(),'status':'running' if any(j['status'] in ('queued','running') for j in jobs) else ('complete' if all(j['status']=='complete' for j in jobs) else 'failed'),'source_sha256':m['source_sha256'],'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'jobs':jobs})
    save()
    while any(j['status'] in ('queued','running') for j in jobs):
        for cpu,process in list(active.items()):
            rc=process.poll()
            if rc is not None:
                job=next(j for j in jobs if j.get('pid')==process.pid);output=ROOT/job['output']
                valid=(rc==0 and (output/'exit.json').exists() and json.loads((output/'exit.json').read_text())['returncode']==0 and (output/'summary.json').exists())
                job.update(status='complete' if valid else 'failed',returncode=rc,finished=time.time());del active[cpu];save()
        for cpu in (24,25):
            if cpu in active:continue
            job=next((j for j in jobs if j['status']=='queued'),None)
            if job is None:continue
            if (ROOT/job['output']).exists():raise FileExistsError(job['output'])
            stream=(args.runtime/f"{job['id']}.log").open('x');handles.append(stream)
            cmd=['taskset','-c',str(cpu),sys.executable,'analysis/run_fixed_endpoint_repair.py','--manifest',str(args.manifest),'--config',job['config'],'--output',job['output']]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            job.update(status='running',pid=process.pid,cpu=cpu,started=time.time(),command=cmd);active[cpu]=process;save()
        if active:time.sleep(2)
    save()
    for handle in handles:handle.close()
    return 0 if all(j['status']=='complete' for j in jobs) else 1


if __name__=='__main__':raise SystemExit(main())
