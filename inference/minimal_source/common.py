from pathlib import Path
import json
p=json.loads((Path(__file__).resolve().parents[1]/'PTC.json').read_text(encoding='utf-8'))
G=p['gain_dn_per_e']
RN=p['analog_read_noise_dn_rms']
K=p['pg_negative_K']
