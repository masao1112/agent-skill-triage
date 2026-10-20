"""UI behavior only; fake predictions are never model-evaluation measurements."""
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from app import App
from classifier import Prediction
from config import Settings


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Display unavailable: {exc}')
        self.root.withdraw()
        self.app = App(self.root, Settings())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'SKILL.md'
        self.path.write_text('a small skill', encoding='utf-8')
        self.app.path.set(str(self.path))

    def tearDown(self):
        if hasattr(self, 'app') and not self.app.closed:
            self.app.close()

    def finish(self):
        deadline = time.monotonic() + 5
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.02)
        self.assertFalse(self.app.busy)

    def test_success_clear_and_reuse(self):
        class FakeClassifier:
            calls = 0
            load_seconds = .1
            def classify(self, _):
                self.calls += 1
                return Prediction('HIGH_RISK', 'HIGH_RISK', 10, 20, .2, .01)
        clf = FakeClassifier()
        self.app.classifier = clf
        self.app.analyze()
        self.finish()
        self.assertEqual(self.app.result.get(), 'HIGH_RISK')
        self.assertIsNotNone(self.app.last_result)
        self.app.clear()
        self.assertIsNone(self.app.last_result)
        self.assertEqual(self.app.result.get(), 'Chưa phân tích')
        self.app.analyze()
        self.finish()
        self.assertEqual(clf.calls, 2)
        self.assertEqual(str(self.app.analyze_button['state']), 'normal')

    def test_error_removes_previous_result(self):
        class FailingClassifier:
            def classify(self, _):
                raise RuntimeError('invalid output')
        self.app.last_result = {'label': 'NORMAL'}
        self.app.classifier = FailingClassifier()
        self.app.analyze()
        self.finish()
        self.assertEqual(self.app.result.get(), 'INFERENCE_ERROR')
        self.assertIsNone(self.app.last_result)
        self.assertEqual(str(self.app.save_button['state']), 'disabled')
        self.assertEqual(str(self.app.analyze_button['state']), 'normal')


    def test_preview_of_downloaded_folder_does_not_load_model(self):
        class MustNotLoad:
            def classify(self, _):
                raise AssertionError('Preview must not call classifier')
        self.app.classifier = MustNotLoad()
        self.app.path.set(str(self.path.parent))
        self.app.analyze(preview=True)
        self.finish()
        self.assertEqual(self.app.result.get(), 'Đã đọc đầu vào')
        self.assertIsNone(self.app.last_result)
        self.assertIn('### FILE: SKILL.md ###', self.app.details.get('1.0', 'end'))
        self.assertEqual(str(self.app.preview_button['state']), 'normal')
