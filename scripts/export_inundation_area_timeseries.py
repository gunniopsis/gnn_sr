"""Export small hourly inundation-area tables from existing Gadi predictions.

No inference or training. Uses the common authors' fold-1 scoring mask.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stream_array(stack, path, key, shape):
    z = stack.enter_context(zipfile.ZipFile(path))
    f = stack.enter_context(z.open(key + '.npy'))
    version = np.lib.format.read_magic(f)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    found, fortran, dtype = readers[version](f)
    if found != shape or fortran or dtype != np.dtype('float32'):
        raise ValueError(f'Unexpected array layout: {path}:{key}')
    return f


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    run = root / 'outputs/v2/full_domain/c2_f2/fold_1/seed_42'
    comparison = run / 'lsg_wet_mask_comparison'
    lsg = root / 'outputs/v2/lsg_full_domain/burnett/fold_1/seed_42'
    manifests = [json.loads((p / 'completion.json').read_text()) for p in (comparison, lsg)]
    if any(m['state'] != 'complete' or m['events'] != 18 for m in manifests):
        raise ValueError('Both evaluations must be complete')
    geometry = comparison / 'mapping_geometry.npz'
    if digest(geometry) != manifests[0]['artifacts'][geometry.name]:
        raise ValueError('Geometry hash mismatch')
    with np.load(geometry) as g:
        ids, mask = g['fine_global_ids'], g['scoring_mask']
        if not np.array_equal(np.sort(ids[mask]), np.sort(g['lsg_wet_global_ids'])):
            raise ValueError('Scoring mask mismatch')
    with np.load(lsg / 'geometry.npz') as g:
        if not np.array_equal(ids, g['fine_global_ids']) or not np.array_equal(mask, g['scoring_mask']):
            raise ValueError('LSG geometry differs')
        area = g['area'][mask].astype(np.float64)
    if len(area) != 769709 or not np.all(area == 400):
        raise ValueError('Unexpected scoring cells or areas')
    with (root / 'reference_files/BurnettRV/BurnettRV_event_summary.csv').open(newline='') as f:
        events = [r['tuflow_evt_name_old'] for r in csv.DictReader(f) if r['Group'] == '1']
    times = np.arange(48, 431)
    hourly, peaks = [], []
    models = ['HF_TUFLOW', 'LF_HECRAS', 'GNN_SR', 'LSG']
    for number, event in enumerate(events, 1):
        prediction = run / 'predictions' / f'{event}.npz'
        lf_prediction = comparison / 'lf_predictions' / f'{event}.npz'
        for p in (prediction, lf_prediction):
            with np.load(p) as d:
                if not np.array_equal(d['fine_global_ids'], ids):
                    raise ValueError(f'Column mismatch: {p}')
                if p == lf_prediction and not np.allclose(d['time_hours'], times, rtol=0, atol=1e-8):
                    raise ValueError('LF time mismatch')
        counts = {model: [] for model in models}
        with ExitStack() as stack:
            shape = (len(times), len(ids))
            streams = [stream_array(stack, p, k, shape) for p,k in
                       [(prediction,'target_depth'), (lf_prediction,'lf_depth'), (prediction,'predicted_depth')]]
            h = stack.enter_context(h5py.File(lsg / 'predictions' / f'{event}.h5', 'r'))
            if not np.array_equal(h['time_hours'][:], times) or h['predicted_depth'].shape != shape:
                raise ValueError('LSG time/shape mismatch; column IDs are checked in geometry.npz')
            for start in range(0, len(times), 8):
                stop = min(start+8, len(times))
                hf = None
                for model, stream in zip(models[:3], streams):
                    size = (stop-start)*len(ids)*4
                    raw = stream.read(size)
                    if len(raw) != size:
                        raise ValueError('Truncated depth array')
                    depth = np.frombuffer(raw, dtype=np.float32).reshape(stop-start,len(ids))[:,mask]
                    if not np.isfinite(depth).all() or np.any(depth < 0):
                        raise ValueError('Invalid physical depths')
                    counts[model].extend(np.count_nonzero(depth >= .03,axis=1).tolist())
                    if model == 'HF_TUFLOW':
                        hf = depth
                if not np.array_equal(h['target_depth'][start:stop, :][:, mask], hf):
                    raise ValueError('LSG and GNN reference depths differ')
                depth = h['predicted_depth'][start:stop, :][:, mask]
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError('Invalid LSG depth')
                counts['LSG'].extend(np.count_nonzero(depth >= .03,axis=1).tolist())
            if any(stream.read(1) for stream in streams):
                raise ValueError('Unexpected array bytes')
        row = {'test_event_number':number,'event':event}
        for model in models:
            values = np.array(counts[model])
            tied = times[values == values.max()]
            row[model+'_time_h'] = int(tied[0])
            row[model+'_tied_peak_times_h'] = ';'.join(map(str,tied))
        for model in models[1:]:
            row[model+'_dt_h'] = row[model+'_time_h'] - row['HF_TUFLOW_time_h']
        peaks.append(row)
        for i,time in enumerate(times):
            record={'test_event_number':number,'event':event,'time_hours':int(time)}
            for model in models:
                record[model+'_wet_cells']=counts[model][i]
                record[model+'_area_km2']=counts[model][i]*400/1e6
            hourly.append(record)
        print(f'Completed {number}/18: {event}',flush=True)
    out = comparison / 'inundation_area_timing'
    out.mkdir(exist_ok=True)
    write_csv(out/'hourly_inundation_area.csv',hourly)
    write_csv(out/'peak_inundation_area_timing.csv',peaks)
    metadata={'mask':'Common LSG fold-1 wet mask, 769709 cells', 'threshold_m':.03,
              'time':'Elapsed simulation hours 48-430, excluding warm-up',
              'peak':'Maximum simultaneous flooded area; earliest hour if tied; all ties recorded',
              'dt':'Model time minus HF TUFLOW time; positive means later',
              'source_completion_sha256':{str(p.relative_to(root)):digest(p/'completion.json') for p in (comparison,lsg)},
              'artifacts':{p.name:digest(p) for p in out.glob('*.csv')}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    main(args.root)
