import json
from pathlib import Path
import tempfile
import unittest
from import_samples import import_samples


class SampleImportTests(unittest.TestCase):
    def test_preserves_source_and_skips_token_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            bad = source / 'G1_normal/bad'
            good = source / 'G1_normal/good'
            bad.mkdir(parents=True); good.mkdir(parents=True)
            (bad / 'SKILL.md').write_text('overflow')
            original = 'normal text ' * 30
            (good / 'SKILL.md').write_text(original)
            def tokenizer(text, **kwargs):
                self.assertIs(kwargs['truncation'], False)
                return {'input_ids': [1] * (10001 if 'overflow' in text else 5)}
            output = root / 'samples'
            imported = import_samples(source, output, groups=['G1_normal'], tokenizer=tokenizer)
            self.assertEqual(imported, [{'path':'G1_normal/good', 'expected_label':'NORMAL'}])
            self.assertEqual((good / 'SKILL.md').read_text(), original)
            self.assertEqual((output / 'G1_normal/good/SKILL.md').read_text(), original)
            self.assertTrue((bad / 'SKILL.md').exists())
            self.assertFalse((output / 'G1_normal/bad').exists())
            provenance = json.loads((output / 'provenance.json').read_text())
            self.assertIn('SKILL.md', provenance[0]['sha256'])

    def test_symlink_package_is_not_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / 'source/G1_normal/bad'
            good = root / 'source/G1_normal/good'
            bad.mkdir(parents=True); good.mkdir(parents=True)
            (bad / 'SKILL.md').write_text('x')
            (bad / 'external.md').symlink_to(root / 'private')
            (good / 'SKILL.md').write_text('safe sample')
            imported = import_samples(root / 'source', root / 'out', groups=['G1_normal'])
            self.assertEqual(imported[0]['path'], 'G1_normal/good')
            self.assertFalse((root / 'out/G1_normal/bad').exists())
