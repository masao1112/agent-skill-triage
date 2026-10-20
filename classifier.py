"""Offline causal-LM inference with an explicit deployment contract."""
import hashlib
import json
import string
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from config import Limits, Settings
from parser import InputError


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    raw_output: str
    package_tokens: int
    input_tokens: int
    inference_seconds: float
    tokenize_seconds: float

    def to_dict(self):
        return asdict(self)


def read_contract(settings):
    metadata = json.loads((settings.adapter / 'adapter_config.json').read_text())
    contract = json.loads(settings.contract.read_text(encoding='utf-8'))
    if contract.get('verified_from_original_pipeline') is not True and contract.get('deployment_mode') != 'new_prompt':
        raise InferenceError('Chưa có hợp đồng inference. Khai báo prompt gốc hoặc deployment_mode=new_prompt.')
    if not contract.get('source'):
        raise InferenceError('Hợp đồng inference thiếu nguồn gốc (source).')
    for key in ('base_model_name_or_path', 'task_type'):
        if contract.get(key) != metadata.get(key):
            raise InferenceError(f'Checkpoint không khớp hợp đồng: {key}.')
    if contract['task_type'] != 'CAUSAL_LM':
        raise InferenceError('Adapter này cần pipeline khác CAUSAL_LM; không được thay đầu phân loại.')
    prompt = contract.get('prompt_template')
    if not isinstance(prompt, str):
        raise InferenceError('Thiếu prompt_template gốc.')
    fields = [(field, spec, conv) for _, field, spec, conv in string.Formatter().parse(prompt) if field is not None]
    if fields != [('text', '', None)]:
        raise InferenceError('prompt_template phải có đúng một placeholder {text}. Escape các ngoặc khác bằng {{ và }}.')
    if type(contract.get('add_special_tokens')) is not bool:
        raise InferenceError('Phải khai báo add_special_tokens từ pipeline gốc.')
    mapping = contract.get('output_labels')
    if not isinstance(mapping, dict) or not mapping or set(mapping.values()) != {'NORMAL', 'HIGH_RISK'}:
        raise InferenceError('output_labels phải ánh xạ chính xác đầu ra gốc sang hai nhãn.')
    if any(not isinstance(k, str) or k != k.strip() or not k for k in mapping):
        raise InferenceError('Khóa output_labels phải là chuỗi không rỗng, không có khoảng trắng ở biên.')
    generation = contract.get('generation')
    if not isinstance(generation, dict) or generation.get('do_sample') is not False:
        raise InferenceError('Cần generation gốc với do_sample=false để đo lặp lại.')
    if type(generation.get('max_new_tokens')) is not int or not 1 <= generation['max_new_tokens'] <= 128:
        raise InferenceError('max_new_tokens phải trong khoảng 1–128.')
    if not isinstance(contract.get('max_context_tokens'), int) or contract['max_context_tokens'] < 1:
        raise InferenceError('Cần max_context_tokens được xác nhận từ pipeline gốc.')
    return contract


def map_output(output, mapping):
    # Match the complete configured label, ignoring case and surrounding whitespace.
    # Preserve raw output for reports; prose/substrings and ambiguous aliases fail.
    normalized = output.strip().casefold()
    matches = {label for raw_label, label in mapping.items()
               if raw_label.casefold() == normalized}
    if len(matches) != 1 or not matches.issubset({'NORMAL', 'HIGH_RISK'}):
        raise InferenceError(f'INFERENCE_ERROR: đầu ra không hợp lệ: {output[:160]!r}')
    return matches.pop()


def validate_package_tokens(text, tokenizer):
    count = len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
    if count > Limits().max_tokens:
        raise InputError(f'Đầu vào có {count:,} token, vượt 10.000 token; không tự cắt ngắn.')
    return count


class Classifier:
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.contract = None
        self.load_seconds = None

    def load(self):
        if self.model is not None:
            return
        start = time.perf_counter()
        try:
            contract = read_contract(self.settings)
            if not (self.settings.base_model / 'config.json').is_file():
                raise InferenceError(f'Chưa có base model offline tại {self.settings.base_model}.')
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            if self.settings.device == 'cuda' and not torch.cuda.is_available():
                raise InferenceError('CUDA không khả dụng. Chọn cpu trong settings.json.')
            # Saved tokenizer is part of the checkpoint; never silently use a different one.
            tokenizer = AutoTokenizer.from_pretrained(str(self.settings.adapter), local_files_only=True,
                                                     trust_remote_code=False)
            kwargs = dict(local_files_only=True, trust_remote_code=False,
                          dtype=torch.float32 if self.settings.device == 'cpu' else torch.float16)
            if self.settings.quantization == 'nf4':
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
                kwargs['device_map'] = {'': 'cuda:0'}
            base = AutoModelForCausalLM.from_pretrained(str(self.settings.base_model), **kwargs)
            if self.settings.quantization == 'none':
                base = base.to(self.settings.device)
            model = PeftModel.from_pretrained(base, str(self.settings.adapter), local_files_only=True)
            model.eval()
            self.model, self.tokenizer, self.contract = model, tokenizer, contract
            self.load_seconds = time.perf_counter() - start
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f'Không nạp được mô hình: {exc}') from exc

    def prepare(self, text):
        start = time.perf_counter()
        # No truncation at either validation stage.
        count = validate_package_tokens(text, self.tokenizer)
        prompt = self.contract['prompt_template'].format(text=text)
        inputs = self.tokenizer(prompt, add_special_tokens=self.contract['add_special_tokens'],
                                truncation=False, return_tensors='pt')
        length = inputs['input_ids'].shape[-1]
        reserve = self.contract['generation']['max_new_tokens']
        model_limit = getattr(self.model.config, 'max_position_embeddings', self.contract['max_context_tokens'])
        limit = min(model_limit, self.contract['max_context_tokens'])
        if length + reserve > limit:
            raise InputError(f'Prompt + đầu ra dự trữ vượt context {limit:,} token.')
        return inputs, count, length, time.perf_counter() - start

    def classify(self, text):
        self.load()
        inputs, count, length, tokenize_seconds = self.prepare(text)
        import torch
        inputs = inputs.to(self.settings.device)
        if self.settings.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **self.contract['generation'])
            if self.settings.device == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            raw = self.tokenizer.decode(output[0, length:], skip_special_tokens=True)
            label = map_output(raw, self.contract['output_labels'])
            return Prediction(label, raw, count, length, seconds, tokenize_seconds)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f'INFERENCE_ERROR: {exc}') from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
