from pathlib import Path
import hashlib, json, os, time
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('SADPS_OUTPUT', str(ROOT / 'outputs/reproduction'))).resolve()
DATA = Path(os.environ.get('SADPS_DATA', str(ROOT / 'datasets/space_v2'))).resolve()
CACHE = Path(os.environ.get('SADPS_CACHE', str(ROOT / 'cache'))).resolve()
PTC = json.loads((ROOT/'configs/ptc_s5m2_iso12800_localRG_locked.json').read_text(encoding='utf-8-sig'))
G = PTC['gain_dn_per_e']
RN_TOTAL = PTC['ptc_read_noise_total_dn_rms']
RN = PTC['analog_read_noise_dn_rms']
K = PTC['pg_negative_K']
BLACK = 512
UPPER = PTC['signed_positive_upper']
MATRIX = [[.3485256854664918,.060970508902393306,.08471232558620155], [.08744703447690182,.700774429962975,.21177856954644453], [.02111332704585997,.10411145975775828,.4182996492763548]]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)
def identity(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
