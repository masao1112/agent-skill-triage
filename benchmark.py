"""Measurements only. No training, parameter selection or fabricated metrics."""
import importlib.metadata
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path
from classifier import Classifier, sha256_file
from parser import parse_input


def distribution(values):
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None}
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)]}


class MemorySampler:
    """Process RSS sampled every 10ms, including model loading (approximate peak)."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def run(settings, manifest, warmup=0):
    import psutil
    import torch
    manifest = Path(manifest).resolve()
    items = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(items, list) or not items or warmup < 0:
        raise ValueError('Manifest phải là danh sách mẫu không rỗng; warmup >= 0.')
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise ValueError('Mỗi mẫu phải có path.')
        for key in ('expected_label', 'reference_label'):
            if item.get(key) is not None and item[key] not in ('NORMAL', 'HIGH_RISK'):
                raise ValueError(f'{key} phải là NORMAL hoặc HIGH_RISK.')
    records = []
    clf = Classifier(settings)
    with MemorySampler() as memory:
        if settings.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        clf.load()
        # Optional warmup uses the first manifest input. Fail explicitly if invalid.
        if warmup:
            first = items[0]
            package = parse_input(manifest.parent / first['path'], serialized=first.get('serialized', False))
            for _ in range(warmup):
                clf.classify(package.text)
        for item in items:
            started = time.perf_counter()
            row = dict(item)
            try:
                parse_start = time.perf_counter()
                package = parse_input(manifest.parent / item['path'], serialized=item.get('serialized', False))
                row['parse_seconds'] = time.perf_counter() - parse_start
                pred = clf.classify(package.text)
                row.update(pred.to_dict(), status='ok', skipped=package.skipped)
                row['input_sha256'] = sha256_file(manifest.parent / item['path'])
            except Exception as exc:
                row.update(status='error', error=str(exc))
            row['total_seconds'] = time.perf_counter() - started
            records.append(row)
        gpu_peak = torch.cuda.max_memory_allocated() if settings.device == 'cuda' else None
        gpu_reserved = torch.cuda.max_memory_reserved() if settings.device == 'cuda' else None
    valid = [r for r in records if r['status'] == 'ok']
    refs = [r for r in records if r.get('reference_label') is not None]
    known = [r for r in records if r.get('expected_label') is not None]
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'psutil', 'bitsandbytes'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    model_files = {}
    for directory in (settings.adapter, settings.base_model):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                model_files[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')), cpu_model)
    return {
        'environment': {'os': platform.platform(), 'python': platform.python_version(),
                        'cpu': cpu_model, 'cpu_count': psutil.cpu_count(),
                        'ram_total_bytes': psutil.virtual_memory().total,
                        'gpu': torch.cuda.get_device_name() if settings.device == 'cuda' else None,
                        'device': settings.device, 'quantization': settings.quantization,
                        'versions': versions},
        'contract': clf.contract, 'contract_sha256': sha256_file(settings.contract),
        'model_files': model_files, 'model_disk_bytes': sum(x['bytes'] for x in model_files.values()),
        'deployment_files_bytes': sum(p.lstat().st_size for p in Path(__file__).resolve().parent.rglob('*') if p.is_file()),
        'venv_files_bytes': sum(p.lstat().st_size for p in (Path(__file__).resolve().parent / '.venv').rglob('*') if p.is_file()),
        'manifest_sha256': sha256_file(manifest), 'warmup_runs': warmup,
        'model_load_seconds': clf.load_seconds,
        'ram_baseline_bytes': memory.baseline, 'ram_peak_sampled_bytes': memory.peak,
        'ram_sampling_interval_seconds': .01,
        'vram_peak_allocated_bytes': gpu_peak, 'vram_peak_reserved_bytes': gpu_reserved,
        'attempts': len(records), 'valid_outputs': len(valid), 'errors': len(records) - len(valid),
        'valid_output_rate': len(valid) / len(records),
        'reference_samples': len(refs),
        'reference_match_rate': sum(r.get('label') == r['reference_label'] for r in refs) / len(refs) if refs else None,
        'labelled_samples': len(known),
        'accuracy_all_labelled_attempts': sum(r.get('label') == r['expected_label'] for r in known) / len(known) if known else None,
        'inference_seconds': distribution([r['inference_seconds'] for r in valid]),
        'total_seconds': distribution([r['total_seconds'] for r in valid]),
        'records': records,
    }
