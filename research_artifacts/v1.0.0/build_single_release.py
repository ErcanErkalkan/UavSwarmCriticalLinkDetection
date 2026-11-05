#!/usr/bin/env python3
"""Build one complete paper-version ZIP from the checked-out repository.

Only the frozen paper/S2 payload and the actual checked-out Oct P0.6 code are included.
Historical source code is not mislabeled as the S2 executed engine.
"""
import argparse
import json
from pathlib import Path
import zipfile
from reconstruct import verify, sha256_file, ROOT, REPO


def build(outdir):
    outdir = Path(outdir)
    parts = verify(outdir)
    version = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())['version']
    bundle = outdir / f'ACLD_PaperMatched_{version}.zip'
    source_paths = sorted([p for p in REPO.glob('*.py') if p.is_file()])
    source_paths += sorted([p for p in REPO.glob('*.ipynb') if p.is_file()])
    source_paths += sorted([p for p in REPO.glob('*.ps1') if p.is_file()])
    source_paths += sorted([p for p in REPO.glob('tests/test*.py') if p.is_file()])
    for name in ('README.md','requirements.txt','pyproject.toml'):
        p = REPO/name
        if p.is_file(): source_paths.append(p)
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        z.write(parts['paper_source'],'ACLD_v1.0.0/paper/ACLD_Paper_Source_v1.0.0.zip',compress_type=zipfile.ZIP_STORED)
        z.write(parts['s2_dataset'],'ACLD_v1.0.0/data/ACLD_S2_Zenodo_Dataset_v1.0.zip',compress_type=zipfile.ZIP_STORED)
        z.write(ROOT/'RELEASE_MANIFEST.json','ACLD_v1.0.0/RELEASE_MANIFEST.json')
        z.write(ROOT/'README.md','ACLD_v1.0.0/README.md')
        for p in source_paths:
            z.write(p, 'ACLD_v1.0.0/repository_source/' + str(p.relative_to(REPO)))
    if zipfile.ZipFile(bundle).testzip() is not None:
        raise ValueError('Bundled release archive is corrupt')
    print('PASS',bundle, bundle.stat().st_size, sha256_file(bundle),len(source_paths),'version-locked source paths')
    return bundle

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='dist')
    build(p.parse_args().out)
