"""Local paths and bounded input handling. No remote model downloads."""
from dataclasses import dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
LABELS = {0: 'NORMAL', 1: 'HIGH_RISK'}


@dataclass(frozen=True)
class Limits:
    max_input_bytes: int = 20 * 1024 * 1024
    max_file_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 20 * 1024 * 1024
    max_files: int = 512
    max_ratio: int = 200
    max_tokens: int = 10000


@dataclass(frozen=True)
class Settings:
    adapter: Path = ROOT / 'models/qwen-adapter'
    base_model: Path = ROOT / 'models/base'
    contract: Path = ROOT / 'inference_contract.json'
    device: str = 'cpu'
    quantization: str = 'none'

    @classmethod
    def read(cls, path=ROOT / 'settings.json'):
        path = Path(path).resolve()
        explicit = path != (ROOT / 'settings.json').resolve()
        if explicit and not path.is_file():
            raise FileNotFoundError(f'Không tìm thấy cấu hình: {path}')
        source = path if path.exists() else ROOT / 'settings.example.json'
        raw = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
        for key in ('adapter', 'base_model', 'contract'):
            if key in raw:
                p = Path(raw[key]).expanduser()
                raw[key] = (path.parent / p).resolve() if not p.is_absolute() else p
        if 'contract' not in raw and not (ROOT / 'inference_contract.json').exists():
            raw['contract'] = ROOT / 'inference_contract.example.json'
        elif raw.get('contract') == ROOT / 'inference_contract.json' and not raw['contract'].exists():
            raw['contract'] = ROOT / 'inference_contract.example.json'
        settings = cls(**raw)
        if settings.device not in ('cpu', 'cuda'):
            raise ValueError('device phải là cpu hoặc cuda.')
        if settings.quantization not in ('none', 'nf4'):
            raise ValueError('quantization phải là none hoặc nf4.')
        if settings.quantization == 'nf4' and settings.device != 'cuda':
            raise ValueError('NF4 yêu cầu CUDA trong ứng dụng này.')
        return settings
