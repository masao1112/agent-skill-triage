import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from classifier import Classifier, InferenceError, map_output, read_contract
from config import Limits, Settings
from parser import InputError, parse_input


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def file(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return path

    def archive(self, entries):
        path = self.root / 'test.zip'
        with zipfile.ZipFile(path, 'w') as stream:
            for name, text in entries:
                stream.writestr(name, text)
        return path

    def test_single_exact_serialization(self):
        p = parse_input(self.file('SKILL.md', b'hello\n'))
        self.assertEqual(p.text, '### FILE: SKILL.md ###\nhello\n')

    def test_pre_serialized_unchanged(self):
        text = b'### FILE: SKILL.md ###\nhello\n\n### FILE: script.py ###\nprint(1)'
        self.assertEqual(parse_input(self.file('full.txt', text), serialized=True).text, text.decode())

    def test_zip_wrapper_and_manifest_order(self):
        path = self.archive([('skill/scripts/z.py', 'z'), ('skill/SKILL.md', 'hello'), ('skill/references/a.md', 'a')])
        p = parse_input(path)
        self.assertEqual(p.text, '### FILE: SKILL.md ###\nhello\n\n### FILE: scripts/z.py ###\nz\n\n### FILE: references/a.md ###\na')

    def test_no_script_execution(self):
        target = self.root / 'EXECUTED'
        path = self.archive([('SKILL.md', 'x'), ('script.py', f'open({str(target)!r}, "w").write("oops")')])
        parse_input(path)
        self.assertFalse(target.exists())

    def test_zip_paths_rejected(self):
        for name in ('../escape', '/tmp/escape', 'x/../../escape', 'C:/escape', 'x\\y', 'bad\npath', '###.txt'):
            with self.subTest(name=name), self.assertRaises(InputError):
                parse_input(self.archive([('SKILL.md', 'x'), (name, 'x')]))

    def test_symlink_rejected(self):
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(InputError):
            parse_input(self.archive([('SKILL.md', 'x'), (info, '/etc/passwd')]))

    def test_duplicates_rejected(self):
        with self.assertWarns(UserWarning):
            path = self.archive([('SKILL.md', 'x'), ('SKILL.md', 'y')])
        with self.assertRaises(InputError):
            parse_input(path)

    def test_empty_invalid_encoding_and_binary(self):
        for content in (b'', b'   \n', b'\xff\xfe', b'a\x00b'):
            with self.subTest(content=content), self.assertRaises(InputError):
                parse_input(self.file('bad.txt', content))

    def test_bad_and_empty_zip(self):
        for path in (self.file('bad.zip', b'not zip'), self.archive([])):
            with self.assertRaises(InputError):
                parse_input(path)

    def test_missing_file(self):
        with self.assertRaises(InputError):
            parse_input(self.root / 'missing')

    def test_zip_without_skill(self):
        with self.assertRaises(InputError):
            parse_input(self.archive([('other.md', 'x')]))

    def test_size_limits(self):
        with self.assertRaises(InputError):
            parse_input(self.file('large.txt', b'0123456789'), limits=Limits(max_file_bytes=5))
        with self.assertRaises(InputError):
            parse_input(self.archive([('SKILL.md', 'x' * 10)]), limits=Limits(max_total_bytes=5))
        with self.assertRaises(InputError):
            parse_input(self.archive([('SKILL.md', 'x'), ('a.py', 'y')]), limits=Limits(max_files=1))

    def test_zip_bomb_ratio(self):
        path = self.root / 'bomb.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr('SKILL.md', 'a' * 100000)
        with self.assertRaises(InputError):
            parse_input(path)

    def test_skipped_files_visible(self):
        p = parse_input(self.archive([('SKILL.md', 'x'), ('image.png', b'\xff'), ('.env', 'SECRET')]))
        self.assertEqual(p.skipped, ('image.png', '.env'))

    def test_invalid_output_never_normal(self):
        mapping = {'NORMAL': 'NORMAL', 'HIGH_RISK': 'HIGH_RISK'}
        self.assertEqual(map_output(' HIGH_RISK\n', mapping), 'HIGH_RISK')
        for output in ('', 'not NORMAL', 'NORMAL HIGH_RISK', '0'):
            with self.subTest(output=output), self.assertRaises(InferenceError):
                map_output(output, mapping)

    def test_case_insensitive_complete_labels(self):
        mapping = {'NORMAL': 'NORMAL', 'HIGH_RISK': 'HIGH_RISK', 'MALICIOUS': 'HIGH_RISK'}
        for raw, expected in [('malicious', 'HIGH_RISK'), (' Malicious\n', 'HIGH_RISK'),
                              ('mALiCioUs', 'HIGH_RISK'), ('normal', 'NORMAL'),
                              (' high_risk\n', 'HIGH_RISK')]:
            with self.subTest(raw=raw):
                self.assertEqual(map_output(raw, mapping), expected)
        for raw in ('not malicious', 'malicious instructions', 'malicious.',
                    'NORMAL or MALICIOUS', 'benign', ''):
            with self.subTest(raw=raw), self.assertRaises(InferenceError):
                map_output(raw, mapping)

    def test_conflicting_case_aliases_are_rejected(self):
        with self.assertRaises(InferenceError):
            map_output('malicious', {'MALICIOUS':'HIGH_RISK', 'malicious':'NORMAL'})

    def test_unverified_contract_refused(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps({'task_type': 'CAUSAL_LM'}))
        contract = self.root / 'contract.json'
        contract.write_text('{"verified_from_original_pipeline": false}')
        with self.assertRaises(InferenceError):
            read_contract(Settings(adapter=adapter, contract=contract))

    def test_explicit_new_prompt_contract_allowed(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        meta = {'task_type': 'CAUSAL_LM', 'base_model_name_or_path': 'Qwen/Qwen3-0.6B'}
        (adapter / 'adapter_config.json').write_text(json.dumps(meta))
        contract = self.root / 'contract.json'
        data = dict(meta, verified_from_original_pipeline=False, deployment_mode='new_prompt',
                    source='new deployment, explicitly authorized', prompt_template='{text}',
                    add_special_tokens=False, max_context_tokens=32768,
                    generation={'do_sample': False, 'max_new_tokens': 16},
                    output_labels={'NORMAL':'NORMAL', 'HIGH_RISK':'HIGH_RISK'})
        contract.write_text(json.dumps(data))
        self.assertEqual(read_contract(Settings(adapter=adapter, contract=contract))['deployment_mode'], 'new_prompt')
        data['base_model_name_or_path'] = 'wrong/base'
        contract.write_text(json.dumps(data))
        with self.assertRaises(InferenceError):
            read_contract(Settings(adapter=adapter, contract=contract))

    def test_token_overflow_never_truncated(self):
        clf = Classifier()
        calls = []
        def tokenizer(text, **kwargs):
            calls.append(kwargs)
            return {'input_ids': list(range(10001))}
        clf.tokenizer = tokenizer
        with self.assertRaises(InputError):
            clf.prepare('oversized')
        self.assertIs(calls[0]['truncation'], False)

    def test_context_includes_reserved_output(self):
        clf = Classifier()
        clf.contract = {'prompt_template': '{text}', 'add_special_tokens': True,
                        'generation': {'max_new_tokens': 10}, 'max_context_tokens': 20}
        clf.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=20))
        def tokenizer(text, **kwargs):
            return {'input_ids': SimpleNamespace(shape=(1, 15))} if 'return_tensors' in kwargs else {'input_ids': [1] * 15}
        clf.tokenizer = tokenizer
        with self.assertRaises(InputError):
            clf.prepare('text')


if __name__ == '__main__':
    unittest.main()
