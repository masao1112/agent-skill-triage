"""Fixed synthetic input experiment, using the actual checkpoint tokenizer."""
import datetime
import json
from pathlib import Path
import tempfile
import zipfile
from classifier import validate_package_tokens
from config import ROOT, Settings
from parser import parse_input, InputError


def run():
    from transformers import AutoTokenizer
    settings = Settings.read()
    tokenizer = AutoTokenizer.from_pretrained(settings.adapter, local_files_only=True, trust_remote_code=False)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, data, expected_error in [
            ('valid.md', b'hello world', False),
            ('empty.md', b'', True), ('whitespace.md', b' \n ', True),
            ('encoding.md', b'\xff\xfe', True), ('binary.md', b'a\x00b', True),
            ('invalid.zip', b'not a zip archive', True),
            ('large.md', b'x' * (2 * 1024 * 1024 + 1), True),
            ('token_overflow.md', ('hello ' * 10020).encode(), True),
        ]:
            path = root / name
            path.write_bytes(data)
            cases.append((name, path, expected_error))
        cases.append(('missing', root / 'does_not_exist', True))
        for name, entries, expected_error in [
            ('valid.zip', [('SKILL.md', 'hello'), ('scripts/a.py', 'print(1)')], False),
            ('empty.zip', [], True),
            ('traversal.zip', [('SKILL.md', 'hello'), ('../escape.py', 'x')], True),
            ('no_skill.zip', [('readme.md', 'hello')], True),
        ]:
            path = root / name
            with zipfile.ZipFile(path, 'w') as archive:
                for filename, text in entries:
                    archive.writestr(filename, text)
            cases.append((name, path, expected_error))
        rows = []
        for name, path, expected_error in cases:
            row = {'case': name, 'expected_error': expected_error}
            try:
                package = parse_input(path)
                row.update(status='ok', tokens=validate_package_tokens(package.text, tokenizer))
            except InputError as exc:
                row.update(status='detected_input_error', error=str(exc))
            except Exception as exc:
                row.update(status='unexpected_exception', error=str(exc))
            row['passed'] = row['status'] == ('detected_input_error' if expected_error else 'ok')
            rows.append(row)
    invalid = [r for r in rows if r['expected_error']]
    valid = [r for r in rows if not r['expected_error']]
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Synthetic parser + actual tokenizer validation only; no model inference',
            'n': len(rows), 'invalid_inputs': len(invalid), 'valid_inputs': len(valid),
            'error_detection_rate': sum(r['status'] == 'detected_input_error' for r in invalid) / len(invalid),
            'valid_input_success_rate': sum(r['status'] == 'ok' for r in valid) / len(valid),
            'unexpected_exceptions': sum(r['status'] == 'unexpected_exception' for r in rows),
            'records': rows}


if __name__ == '__main__':
    report = run()
    target = ROOT / 'reports/robustness.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, ensure_ascii=False, indent=2))
    raise SystemExit(not all(row['passed'] for row in report['records']))
