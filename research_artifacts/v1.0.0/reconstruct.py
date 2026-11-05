#!/usr/bin/env python3
"""Reassemble and verify the exact author-frozen paper and synthetic S2 archive.

This validates data and manuscript byte identity; it does not regenerate simulations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST = ROOT / 'RELEASE_MANIFEST.json'


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def verify(outdir):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = {}
    for key, spec in manifest['archives'].items():
        chunk_dir = ROOT / spec['parts_directory']
        parts = sorted(chunk_dir.glob('part_*.bin'))
        if len(parts) != spec['parts_count']:
            raise ValueError(f'{key}: expected {spec["parts_count"]} chunks; got {len(parts)}')
        expected_names = [f'part_{i:02d}.bin' for i in range(spec['parts_count'])]
        if [p.name for p in parts] != expected_names:
            raise ValueError(f'{key}: missing, extra or out-of-order chunk')
        output = outdir / spec['filename']
        with output.open('wb') as dest:
            for part in parts:
                with part.open('rb') as source:
                    while True:
                        raw = source.read(1024 * 1024)
                        if not raw:
                            break
                        dest.write(raw)
        actual = sha256_file(output)
        if actual != spec['sha256'] or output.stat().st_size != spec['size_bytes']:
            raise ValueError(f'{key}: SHA-256 / byte-count mismatch; got {actual}')
        with zipfile.ZipFile(output) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'{key}: corrupt ZIP member {bad}')
        names[key] = output
        print(f'PASS {key}: {actual} ({output.stat().st_size} bytes)')

    with zipfile.ZipFile(names['paper_source']) as z:
        for name, digest in manifest['paper_internal_sha256'].items():
            actual = sha256_bytes(z.read('paper/' + name))
            if actual != digest:
                raise ValueError(f'manuscript member {name}: {actual} != {digest}')
            plain = ROOT / 'manuscript' / name
            if plain.exists() and sha256_file(plain) != actual:
                raise ValueError(f'Git-tracked manuscript {plain} differs from paper archive')
    with zipfile.ZipFile(names['s2_dataset']) as z:
        csvs = sorted(n for n in z.namelist() if '/data/units/' in n and n.endswith('.csv'))
        if len(csvs) != 40:
            raise ValueError(f'Expected 40 raw S2 CSVs, got {len(csvs)}')
        for path in csvs:
            meta = json.loads(z.read(path[:-4] + '.json'))
            if sha256_bytes(z.read(path)) != meta['sha256']:
                raise ValueError(f'S2 sidecar digest mismatch: {path}')
        metrics = json.loads(z.read('ACLD_S2_Zenodo_Dataset_v1.0/reanalysis/S2_onset_verified_summary.json'))
        a = metrics['counts_and_metrics']['acld_v3']
        t = metrics['counts_and_metrics']['tarjan_predict_guard']
        if (metrics['verified_unit_sha256'], a['TP'], a['FP'], t['TP'], t['FP']) != (40, 45, 117, 46, 712):
            raise ValueError('S2 scientific metadata does not match the frozen analysis')
    print('PASS 40/40 S2 unit SHA sidecars, frozen LaTeX, bibliography and PDF identities')
    return names


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='dist')
    args = p.parse_args()
    verify(args.out)
