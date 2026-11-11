"""Verify hourly areas and produce the appendix peak-area timing table."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/v2/full_domain/c2_f2/fold_1/seed_42/inundation_area_timing'
MODELS = ['HF_TUFLOW', 'LF_HECRAS', 'GNN_SR', 'LSG']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source):
    metadata = json.loads((source/'metadata.json').read_text())
    for name, expected in metadata['artifacts'].items():
        if digest(source/name) != expected:
            raise ValueError(f'Transfer checksum mismatch: {name}')
    for directory, expected in metadata['source_completion_sha256'].items():
        if digest(ROOT/directory/'completion.json') != expected:
            raise ValueError(f'Source evaluation identity mismatch: {directory}')
    with (source/'peak_inundation_area_timing.csv').open(newline='') as f:
        peaks = list(csv.DictReader(f))
    with (source/'hourly_inundation_area.csv').open(newline='') as f:
        hourly = list(csv.DictReader(f))
    with (ROOT/'reference_files/BurnettRV/BurnettRV_event_summary.csv').open(newline='') as f:
        events = [r['tuflow_evt_name_old'] for r in csv.DictReader(f) if r['Group']=='1']
    if len(peaks)!=18 or len(hourly)!=18*383:
        raise ValueError('Unexpected row counts')
    table, ties = [], []
    peaks.sort(key=lambda r:int(r['test_event_number']))
    for i, (peak,event) in enumerate(zip(peaks,events),1):
        if int(peak['test_event_number'])!=i or peak['event']!=event:
            raise ValueError('Test numbering mismatch')
        series=sorted((r for r in hourly if r['event']==event),key=lambda r:int(r['time_hours']))
        if [int(r['time_hours']) for r in series]!=list(range(48,431)):
            raise ValueError('Missing or duplicate hourly data')
        if any(int(r['test_event_number'])!=i for r in series):
            raise ValueError('Hourly event numbering mismatch')
        for model in MODELS:
            counts=[int(r[model+'_wet_cells']) for r in series]
            if any(n<0 or n>769709 for n in counts):
                raise ValueError('Invalid wet counts')
            if any(abs(float(r[model+'_area_km2'])-n*.0004)>1e-9 for r,n in zip(series,counts)):
                raise ValueError('Area/count mismatch')
            peak_times=[int(r['time_hours']) for r,n in zip(series,counts) if n==max(counts)]
            if int(peak[model+'_time_h'])!=peak_times[0] or peak[model+'_tied_peak_times_h']!=';'.join(map(str,peak_times)):
                raise ValueError('Exported peak timing disagrees with hourly areas')
            if len(peak_times)>1:
                ties.append({'event':event,'model':model,'times':peak_times})
        deltas=[int(peak[m+'_time_h'])-int(peak['HF_TUFLOW_time_h']) for m in MODELS[1:]]
        if deltas!=[int(peak[m+'_dt_h']) for m in MODELS[1:]]:
            raise ValueError('Incorrect time difference')
        table.append([str(i)]+[peak[m+'_time_h'] for m in MODELS]+[f'{d:+d}' if d else '0' for d in deltas])
    header=['Test event no.','HF TUFLOW (h)','LF HEC-RAS (h)','GNN-SR (h)','LSG (h)',
            'LF HEC-RAS Δt (h)','GNN-SR Δt (h)','LSG Δt (h)']
    caption='Time since simulation start to reach maximum total inundation area for the 18 held-out test events.'
    note='Times are in hours from simulation start, evaluated at hourly intervals over hours 48–430. Inundation is defined by depth ≥0.03 m on the common 769,709-cell fold 1 wet mask. Δt = model peak-area time − HF TUFLOW peak-area time; positive values indicate a later peak and negative values an earlier peak. Maximum area is the largest simultaneous inundated area, not the accumulated maximum flood extent. All 72 event/model area series have a unique maximum.'
    if ties:
        note=note.replace('All 72 event/model area series have a unique maximum.','For tied maxima, the earliest hour is reported; ties are recorded in the verification JSON.')
    md=[f'**{caption}**','', '| '+' | '.join(header)+' |', '|'+ '---:|'*len(header)]
    md+=['| '+' | '.join(row)+' |' for row in table]
    md+=['',note]
    markup='<!doctype html><html><head><meta charset="utf-8"><title>Peak inundation-area timing</title><style>body{font-family:Arial,sans-serif;margin:25px}table{border-collapse:collapse;font-size:11pt}th,td{border:1px solid black;padding:7px;text-align:center}th{font-weight:normal}caption{text-align:left;margin-bottom:12px}p{font-size:10pt;max-width:1050px}@media print{@page{size:A4 landscape;margin:15mm}}</style></head><body>'
    markup+='<table><caption>'+caption+'</caption><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in header)+'</tr></thead><tbody>'
    markup+=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table)
    markup+='</tbody></table><p>'+html.escape(note)+'</p></body></html>'
    out=ROOT/'reports/tables/burnett_fold1_peak_inundation_area_timing'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    out.with_suffix('.html').write_text(markup,encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps({'source':str(source),'metadata_sha256':digest(source/'metadata.json'),
        'artifacts':metadata['artifacts'],'verified_events':18,'verified_hourly_rows':len(hourly),'tied_maxima':ties,
        'verification':'Transfer checksums and source identities; catalogue numbering; all hourly areas, peak times, ties and time differences verified'},indent=2)+'\n')
    print(out.with_suffix('.html'))
    print(f'Verified {len(peaks)} events and {len(hourly)} hourly rows; {len(ties)} tied maxima.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    main(parser.parse_args().source)
