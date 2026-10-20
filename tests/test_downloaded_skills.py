import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import config
from config import Limits, Settings
from parser import InputError, parse_input


class DownloadedSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / 'downloaded'
        self.skill.mkdir()
        (self.skill / 'SKILL.md').write_text('skill instructions', encoding='utf-8')

    def test_directory_serialization_includes_supporting_files(self):
        (self.skill / 'scripts').mkdir()
        (self.skill / 'scripts/helper.py').write_text('print(1)', encoding='utf-8')
        package = parse_input(self.skill)
        self.assertEqual(package.files, ('SKILL.md', 'scripts/helper.py'))
        self.assertEqual(package.text, '### FILE: SKILL.md ###\nskill instructions\n\n### FILE: scripts/helper.py ###\nprint(1)')

    def test_source_label_metadata_is_never_model_input(self):
        (self.skill / 'sample_meta.json').write_text('{"label":"jailbreak"}')
        package = parse_input(self.skill)
        self.assertNotIn('jailbreak', package.text)
        self.assertIn('sample_meta.json', package.skipped)
        archive = self.root / 'skill.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('SKILL.md', 'skill instructions')
            stream.writestr('sample_meta.json', '{"label":"jailbreak"}')
        self.assertEqual(parse_input(archive).text, package.text)
        self.assertIn('sample_meta.json', parse_input(archive).skipped)

    def test_file_and_directory_symlinks_rejected(self):
        target = self.root / 'outside.txt'
        target.write_text('private text')
        link = self.skill / 'link.txt'
        link.symlink_to(target)
        with self.assertRaises(InputError):
            parse_input(self.skill)
        with self.assertRaises(InputError):
            parse_input(link)
        link.unlink()
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(InputError):
            parse_input(self.skill)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO only')
    def test_special_file_rejected_without_opening(self):
        os.mkfifo(self.skill / 'pipe.txt')
        with self.assertRaises(InputError):
            parse_input(self.skill)

    def test_directory_limits_and_encoding(self):
        with self.assertRaises(InputError):
            parse_input(self.skill, limits=Limits(max_total_bytes=2))
        (self.skill / 'bad.md').write_bytes(b'\xff')
        with self.assertRaises(InputError):
            parse_input(self.skill)
        with self.assertRaises(InputError):
            parse_input(self.skill, serialized=True)

    def test_clean_clone_configuration_falls_back_to_public_templates(self):
        examples = {
            'adapter': 'models/qwen-adapter', 'base_model': 'models/base',
            'contract': 'inference_contract.json', 'device': 'cpu', 'quantization': 'none',
        }
        (self.root / 'settings.example.json').write_text(json.dumps(examples))
        (self.root / 'inference_contract.example.json').write_text('{}')
        with patch.object(config, 'ROOT', self.root):
            settings = Settings.read(self.root / 'settings.json')
            self.assertEqual(settings.adapter, self.root / 'models/qwen-adapter')
            self.assertEqual(settings.contract, self.root / 'inference_contract.example.json')
            with self.assertRaises(FileNotFoundError):
                Settings.read(self.root / 'typo.json')
