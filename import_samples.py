"""Copy a small, reproducible local sample set. Never execute skill contents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from config import ROOT
from parser import parse_input, InputError

GROUPS = {
    'G1_normal': 'NORMAL',
    'G2_prompt_injection': 'HIGH_RISK',
    'G3_data_exfiltration': 'HIGH_RISK',
    'G4_jailbreak': 'HIGH_RISK',
}


def import_samples(source, output=ROOT / 'samples/local', per_group=1, *, groups=None, tokenizer=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or not 1 <= per_group <= 10:
        raise ValueError('Source phải là thư mục; per_group trong khoảng 1–10.')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Thư mục đích phải tách biệt thư mục nguồn.')
    output.mkdir(parents=True, exist_ok=True)
    manifest, provenance = [], []
    for group in groups or GROUPS:
        if group not in GROUPS:
            raise ValueError(f'Nhóm không hỗ trợ: {group}')
        label = GROUPS[group]
        parent = source / group
        if not parent.is_dir() or parent.is_symlink():
            continue
        candidates = [p for p in parent.iterdir() if p.is_dir() and not p.is_symlink() and (p / 'SKILL.md').is_file()]
        candidates.sort(key=lambda p: (sum(item.stat().st_size for item in p.rglob('*') if item.is_file() and not item.is_symlink()), p.name))
        copied = 0
        for candidate in candidates:
            # These are fixtures, never instructions for the importer itself.
            if any(p.is_symlink() for p in candidate.rglob('*')):
                continue
            try:
                package = parse_input(candidate)
                if tokenizer is not None:
                    from classifier import validate_package_tokens
                    validate_package_tokens(package.text, tokenizer)
            except (InputError, OSError):
                continue
            target = output / group / candidate.name
            if target.exists():
                # Never overwrite modifications to an already imported sample.
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, target, copy_function=shutil.copyfile)
            relative = target.relative_to(output).as_posix()
            manifest.append({'path': relative, 'expected_label': label})
            checksums = {}
            for item in sorted(target.rglob('*')):
                if item.is_file():
                    checksums[item.relative_to(target).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
            provenance.append({'path': relative, 'source': str(candidate), 'source_group': group,
                               'source_label': label, 'sha256': checksums})
            copied += 1
            if copied >= per_group:
                break
    if not manifest:
        raise ValueError('Không có sample mới phù hợp; các thư mục đích đã tồn tại được giữ nguyên.')
    # Preserve the manifest of previous imports as well.
    manifest_path = output / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps(previous + manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    provenance_path = output / 'provenance.json'
    previous_sources = json.loads(provenance_path.read_text()) if provenance_path.exists() else []
    provenance_path.write_text(json.dumps(previous_sources + provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'samples/local')
    p.add_argument('--per-group', type=int, default=1)
    p.add_argument('--group', action='append', choices=tuple(GROUPS), help='Chỉ nhập nhóm này; có thể lặp lại')
    p.add_argument('--validate-tokens', action='store_true', help='Chỉ nhập gói trong giới hạn token bằng tokenizer checkpoint offline')
    args = p.parse_args()
    tokenizer = None
    if args.validate_tokens:
        from transformers import AutoTokenizer
        from config import Settings
        tokenizer = AutoTokenizer.from_pretrained(Settings.read().adapter, local_files_only=True, trust_remote_code=False)
    for item in import_samples(args.source, args.output, args.per_group, groups=args.group, tokenizer=tokenizer):
        print(item['expected_label'], item['path'])


if __name__ == '__main__':
    main()
