"""Offline CLI and reproducible deployment measurements."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from classifier import Classifier, read_contract
from config import ROOT, Settings
from parser import parse_input


def doctor(settings):
    checks = {}
    for name, path in [('adapter', settings.adapter / 'adapter_model.safetensors'),
                       ('tokenizer', settings.adapter / 'tokenizer.json'),
                       ('base_model_config', settings.base_model / 'config.json'),
                       ('base_model_weights', settings.base_model / 'model.safetensors')]:
        checks[name] = {'ok': path.is_file(), 'path': str(path)}
    try:
        read_contract(settings)
        checks['inference_contract'] = {'ok': True}
    except Exception as exc:
        checks['inference_contract'] = {'ok': False, 'error': str(exc)}
    for name in ('torch', 'transformers', 'peft', 'psutil'):
        try:
            checks[name] = {'ok': True, 'version': importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            checks[name] = {'ok': False}
    checks['ready'] = all(v['ok'] for v in checks.values())
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', type=Path, default=ROOT / 'settings.json')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    for name in ('parse', 'analyze'):
        cmd = sub.add_parser(name)
        cmd.add_argument('input', type=Path)
        cmd.add_argument('--serialized', action='store_true')
        if name == 'analyze':
            cmd.add_argument('--output', type=Path, help='Lưu báo cáo JSON (nên đặt trong reports/)')
    bench = sub.add_parser('benchmark')
    bench.add_argument('manifest', type=Path)
    bench.add_argument('--output', type=Path, default=ROOT / 'reports/benchmark.json')
    bench.add_argument('--warmup', type=int, default=0)
    args = p.parse_args()
    try:
        settings = Settings.read(args.settings)
        if args.command == 'doctor':
            report = doctor(settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['ready'] else 2
        if args.command == 'benchmark':
            from benchmark import run
            report = run(settings, args.manifest, warmup=args.warmup)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Đã lưu báo cáo: {args.output.resolve()}')
            return 0
        package = parse_input(args.input, serialized=args.serialized)
        if args.command == 'parse':
            print(package.text)
            print(json.dumps({'files': package.files, 'skipped': package.skipped}, ensure_ascii=False), file=sys.stderr)
        else:
            clf = Classifier(settings)
            prediction = clf.classify(package.text)
            result = dict(input=str(args.input.resolve()), files=package.files, skipped=package.skipped,
                          **prediction.to_dict(), model_load_seconds=clf.load_seconds)
            rendered = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + '\n', encoding='utf-8')
            print(rendered)
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
