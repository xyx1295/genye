"""Copy and verify both archives; derive validation only from the original training pool."""
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/workspace/datasets/genyes8_20261019')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    archives = ROOT / 'archives'
    archives.mkdir(exist_ok=True)
    report = {}
    for kind in ['full', 'lite20']:
        name = f'chushuang2027_3DConveyor_{kind}_s2027'
        source = Path('/shared/datasets') / (name + '.zip')
        copied = archives / source.name
        if not copied.exists():
            shutil.copy2(source, copied)
        digest = sha(source)
        assert sha(copied) == digest, f'ZIP checksum mismatch: {copied}'
        print('ZIP verified:', name, flush=True)
        target = ROOT / name
        marker = ROOT / (name + '.verified.json')
        if not marker.exists():
            with zipfile.ZipFile(copied) as z:
                for info in z.infolist():
                    dest = (ROOT / info.filename).resolve()
                    assert dest.is_relative_to(ROOT.resolve())
                    assert not ((info.external_attr >> 16) & 0o170000) == 0o120000
                z.extractall(ROOT)
            rows = list(csv.DictReader((target / 'split_manifest.csv').open()))
            for i, row in enumerate(rows):
                for key in ['image', 'depth', 'label']:
                    assert sha(target / row[key]) == row[key + '_sha256'], (name, row[key])
                if i % 3000 == 0:
                    print(name, 'verified samples', i, '/', len(rows), flush=True)
            marker.write_text(json.dumps({'archive_sha256': digest, 'verified_samples': len(rows)}, indent=2))
        report[kind] = json.loads(marker.read_text())

    full = ROOT / 'chushuang2027_3DConveyor_full_s2027'
    rows = list(csv.DictReader((full / 'split_manifest.csv').open()))
    by_site = defaultdict(list)
    for row in rows:
        by_site[row['site']].append(row)
    membership = {}
    duplicate_groups = 0
    original_cross_split_groups = 0
    for site, site_rows in by_site.items():
        groups = defaultdict(list)
        for row in site_rows:
            groups[row['image_sha256']].append(row)
        duplicate_groups += sum(len(g) > 1 for g in groups.values())
        original_cross_split_groups += sum(len({r['split'] for r in g}) > 1 for g in groups.values())
        ranked = sorted(groups, key=lambda h: hashlib.sha256(('genyes8-group-2027:' + h).encode()).hexdigest())
        test_target = sum(r['split'] == 'test' for r in site_rows)
        val_target = (len(site_rows) - test_target + 5) // 10
        tests = {h for h, g in groups.items() if any(r['split'] == 'test' for r in g)}
        excess = sum(len(groups[h]) for h in tests) - test_target
        # Keep duplicate groups intact; swap out deterministic singleton test samples.
        for h in ranked:
            if excess and h in tests and len(groups[h]) == 1:
                tests.remove(h)
                excess -= 1
        assert excess == 0
        remaining = val_target
        vals = set()
        for h in ranked:
            if h not in tests and len(groups[h]) <= remaining:
                vals.add(h)
                remaining -= len(groups[h])
        assert remaining == 0
        for h, group in groups.items():
            split = 'test' if h in tests else 'val' if h in vals else 'train'
            for row in group:
                membership[row['sample_id']] = split
    output = ROOT / 'split_63_7_30_grouped'
    output.mkdir(exist_ok=True)
    counts = Counter()
    class_counts = defaultdict(Counter)
    assignments = []
    for row in rows:
        split = membership[row['sample_id']]
        counts[split] += 1
        class_counts[split].update({str(c): int(row[f'class_{c}']) for c in range(4)})
        assignments.append({'sample_id': row['sample_id'], 'site': row['site'], 'split': split})
        for key, dirname in [('image', 'images'), ('depth', 'depth'), ('label', 'labels')]:
            dst = output / dirname / split / Path(row[key]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = full / row[key]
            if not dst.exists():
                dst.symlink_to(src)
            assert dst.resolve() == src.resolve()
    assert len({r['sample_id'] for r in assignments}) == len(rows)
    # Also prevent identical RGB bytes from crossing split boundaries.
    seen = {}
    for row, assignment in zip(rows, assignments):
        h = row['image_sha256']
        if h in seen:
            assert seen[h] == assignment['split'], 'Identical RGB content crosses splits'
        seen[h] = assignment['split']
    with (output / 'split_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'site', 'split'])
        writer.writeheader(); writer.writerows(assignments)
    names = ['0_ok_isolated_complete', '1_ng_related_partial_or_bottom', '2_ng_related_complete_or_top', '3_edge_uncertain_or_truncated']
    config = '\n'.join([f'path: {output}', 'train: images/train', 'val: images/val', 'test: images/test', 'nc: 4', 'names:'] + [f'  {i}: {n}' for i, n in enumerate(names)]) + '\n'
    (output / 'data.yaml').write_text(config)
    report.update(counts=dict(counts), class_counts={k: dict(v) for k, v in class_counts.items()}, split_seed=2027,
                  duplicate_rgb_groups=duplicate_groups, original_cross_train_test_duplicate_groups=original_cross_split_groups,
                  test_membership_changes=sum((r['split']=='test') != (membership[r['sample_id']]=='test') for r in rows),
                  test_policy='Same RGB hash grouped; preserve per-site test counts with deterministic singleton swaps; reserve final test before training', data_yaml=str(output / 'data.yaml'))
    (ROOT / 'preparation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
