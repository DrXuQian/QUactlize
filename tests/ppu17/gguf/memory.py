"""Direct executable memory gate; validate instrumentation with a scalar fault."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def family(pid):
    pairs=[tuple(map(int,line.split())) for line in
           subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True).splitlines()]
    owned={pid}
    while True:
        after=owned|{p for p,parent in pairs if parent in owned}
        if after==owned:return owned
        owned=after

def birth(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--application',type=Path,required=True)
    p.add_argument('--sanitizer',type=Path,required=True)
    p.add_argument('--backend-source',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-uuid',required=True)
    p.add_argument('--configs',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--device',default='0')
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--racecheck-report',choices=('analysis','hazard'),default='analysis')
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh memory output required')
    a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.backend_source))
    import run_benchmark as guard
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    library=a.bundle/manifest['builds']['adapter']['file']
    if sha(library)!=manifest['builds']['adapter']['sha256']:raise ValueError('library differs')
    result={'status':'INCOMPLETE','scope':'DIRECT_HOPPER_SANITIZER_NOT_TIMING',
            'application_sha256':sha(a.application),'library_sha256':sha(library),
            'sanitizer_sha256':sha(a.sanitizer),'runner_sha256':sha(Path(__file__)),'cases':[]}
    retained={}
    if a.resume_from:
        old=json.loads(a.resume_from.read_text())
        for key in ('scope','application_sha256','library_sha256','sanitizer_sha256'):
            if old[key]!=result[key]:raise ValueError('memory resume identity differs: '+key)
        for record in old['cases']:
            if record['status']!='PASS':continue
            if sha(record['log'])!=record['log_sha256']:raise ValueError('memory log differs')
            record['retained_from']=str(a.resume_from)
            key=(record['tool'],record['dtype'],record['config'],record['negative'])
            if key in retained:raise ValueError('duplicate retained memory cell')
            retained[key]=record
    cases=[('memcheck',0,a.configs[0],True)]
    cases += [(tool,d,c,False) for tool in ('memcheck','racecheck','synccheck')
              for d in (0,1) for c in a.configs]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':a.device,
         'LD_LIBRARY_PATH':str(a.bundle)+':'+os.environ.get('LD_LIBRARY_PATH','')}
    for index,(tool,dtype,config,negative) in enumerate(cases):
        if (tool,dtype,config,negative) in retained:
            result['cases'].append(retained[(tool,dtype,config,negative)])
            continue
        log=a.output/f'{index:02d}-{tool}-d{dtype}-c{config}.log'
        command=[str(a.sanitizer),'--target-processes','application-only',
                 '--print-session-details','--error-exitcode','97','--tool',tool,
                 '--check-bulk-copy','yes','--check-warpgroup-mma','yes']
        if tool=='memcheck':command+=['--padding','128']
        if tool=='racecheck':command+=['--racecheck-report',a.racecheck_report]
        command += [str(a.application),str(dtype),str(config)]
        if negative:command+=['scalar-negative']
        record={'tool':tool,'dtype':dtype,'config':config,'negative':negative,'command':command}
        try:
            quiet=0;deadline=time.monotonic()+15
            while quiet<3:
                snap=guard.snapshot(a.device)
                if snap['uuid']!=a.expected_uuid or snap['processes']:
                    guard.require_idle(snap,a.expected_uuid)
                quiet=quiet+1 if not snap['gpu_util'] else 0
                if time.monotonic()>deadline:raise ValueError('GPU did not settle to idle')
                time.sleep(.25)
            with log.open('w') as output:
                child=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic()
                owned={}
                try:
                    while child.poll() is None:
                        for pid in family(child.pid):owned[pid]=birth(pid)
                        snap=guard.snapshot(a.device)
                        for pid in family(child.pid):owned[pid]=birth(pid)
                        # NVML may still report our target after it has exited
                        # and disappeared from ps. Keep proved descendants;
                        # reject any live reused PID with a different birth.
                        foreign=[x for x in snap['processes'] if x['pid'] not in owned or
                                 birth(x['pid']) not in (None,owned[x['pid']])]
                        if snap['uuid']!=a.expected_uuid or foreign:
                            record['foreign_processes']=foreign
                            record['observed_uuid']=snap['uuid']
                            raise ValueError('device changed or foreign job arrived')
                        if time.monotonic()-start>180:raise ValueError('sanitizer timeout')
                        time.sleep(.1)
                finally:
                    if child.poll() is None:
                        # This new session belongs exclusively to our tool and
                        # target child; never signal the foreign GPU process.
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
            text=log.read_text()
            if negative:
                valid=child.returncode!=0 and 'Invalid __global__ write' in text
            else:
                clean='RACECHECK SUMMARY: 0 hazards displayed' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
                valid=child.returncode==0 and clean in text and 'GGUF_Q4_SMOKE PASS' in text
            record.update(status='PASS' if valid else 'FAIL',returncode=child.returncode,log=str(log),log_sha256=sha(log))
        except Exception as error:record.update(status='INVALID',error=str(error))
        result['cases'].append(record)
        result['status']='PASS' if len(result['cases'])==len(cases) and record['status']=='PASS' else 'INCOMPLETE'
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('GGUF_Q4_MEMORY '+json.dumps(record),flush=True)
        if record['status']!='PASS':return 1
    result['status']='PASS'
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
