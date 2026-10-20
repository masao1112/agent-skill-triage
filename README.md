# Agent Skill Triage

Local desktop triage for downloaded AI agent skills, built with Python/Tkinter
and a Qwen3-0.6B LoRA classifier. Select a ZIP, UTF-8 file, or unpacked skill folder;
inspect the parsed content, then classify it as **NORMAL** or **HIGH_RISK**.

The application reads skill packages as data. It never installs skills or executes
their scripts. Inference runs offline; HIGH_RISK means “review before installing,”
and NORMAL is not an assurance of safety. The current deployment prompt uses
NORMAL/MALICIOUS internally and maps MALICIOUS to HIGH_RISK. It is not verified as
the original training prompt; independently evaluate your checkpoint before use.

## Quick start

Python 3.12 with Tkinter is recommended. On Ubuntu, Tkinter is available through
`python3-tk`. Create a virtual environment and install the runtime:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

Prepare the base model once (Internet required for this step):

```bash
.venv/bin/python prepare_model.py
```

The downloader verifies weight SHA256 against Hugging Face metadata; it can use
Hugging Face and the public hf-mirror.com mirror for resumable downloads. It only
transfers public model assets, never input skills. Use `--revision <commit>` to pin
an existing training base revision. The downloaded revision is recorded locally.

**Supply your trained Qwen3-0.6B CAUSAL_LM LoRA adapter and saved tokenizer** under
`models/qwen-adapter/`. Weights are not bundled with the open-source code. The base
model alone is not the trained classifier. Adapter files should include
`adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, and tokenizer
configuration. This application rejects incompatible adapter task/base metadata.

```bash
.venv/bin/python cli.py doctor
./run.sh
```

`run.sh` uses `.venv` and forces offline Hugging Face operation. Subsequent input
parsing and inference require no server or LLM API. Models load once per process.

## Check a skill you just downloaded

```bash
# Select the downloaded package in the GUI; click Preview, then Analyze.
./run.sh "$HOME/Downloads/my-skill.zip"

# Inspect an unpacked skill folder, including scripts/references.
./run.sh "$HOME/Downloads/my-skill"

# Start analysis immediately when explicitly requested.
./run.sh "$HOME/Downloads/my-skill.zip" --analyze

# CLI with a local JSON report.
.venv/bin/python cli.py analyze "$HOME/Downloads/my-skill.zip" --output reports/my-skill.json
```

The file picker opens in Downloads. **Thư mục…** selects a folder containing a
root SKILL.md. **Samples…** opens locally imported fixtures. **Xem nội dung**
previews the exact serialized text plus included/skipped files without loading
models. Preview display is capped at 100,000 characters; model input is validated
separately and never truncated. **Clear** resets the previous result. **Lưu kết quả
JSON** defaults to the ignored `reports/` directory.

Selecting a single SKILL.md scans that file only. Select the whole folder or ZIP
to include supporting files. Inputs already serialized by the research pipeline
can use the full_text checkbox, or `--serialized` on the CLI.

## Configuration

A fresh clone uses `settings.example.json` and `inference_contract.example.json`.
For local overrides, copy them to the ignored filenames:

```bash
cp settings.example.json settings.json
cp inference_contract.example.json inference_contract.json
```

Paths in a settings file are relative to that file. The default adapter path is
`models/qwen-adapter`, the base path is `models/base`. `--settings path.json` accepts
an explicit configuration; a missing explicit file is an error.

CPU uses float32. CUDA requires a suitable PyTorch installation; optional NF4 needs
`requirements-cuda.txt` and settings `device=cuda, quantization=nf4`. Quantization
and precision changes must be independently evaluated. The original training
prompt is not recoverable merely from an adapter or generic chat template.

## Samples and evaluation

Synthetic examples under `examples/` can be shared with the app. Import research
fixtures locally without modifying their originals:

```bash
.venv/bin/python import_samples.py /path/to/raw_samples --per-group 1 --validate-tokens
.venv/bin/python cli.py benchmark samples/local/manifest.json --output reports/local-samples.json
.venv/bin/python robustness.py
python -m unittest discover -s tests -v
```

See [sample handling](samples/README.md), [experiment protocol](EXPERIMENTS.md),
and [Vietnamese usage](docs/usage.vi.md). Source folder labels are annotations;
benchmark accuracy against them is not a claim about real-world maliciousness.

A benchmark manifest is a JSON list of `path` values relative to the manifest.
Optional `expected_label` and `reference_label` use NORMAL/HIGH_RISK;
`reference_label` must come from an independent original inference script.
`serialized: true` preserves an existing full_text input without adding headers.

Reports include individual outcomes, errors, valid output rate, mean/median/p95
latency, process RSS sampled every 10ms, optional peak VRAM, software/hardware,
model/config/input hashes and installation size. Missing references yield null
match metrics. Model load is measured separately; inference measures generation,
while per-sample total includes parsing/tokenization. The benchmark imports torch
before timing load; GUI first initialization may include additional runtime imports.
No held-out accuracy claim is made for the bundled development prompt.

## Input rules

ZIP files are read in memory without extraction. Directory symlinks, file symlinks,
special files, path traversal, duplicate ZIP paths, encrypted ZIPs and malformed
UTF-8 are rejected. Defaults: 512 entries, 2MiB/file, 20MiB/package, ZIP ratio <=200.
Inputs over 10,000 package tokens or the configured prompt/output context limit
are rejected, never cut to force a label. Complete configured output labels are
matched case-insensitively after trimming surrounding whitespace (for example,
`malicious` maps to HIGH_RISK). Prose, substrings and unknown labels remain errors;
raw model output is preserved in reports.

Serialization is `### FILE: relative/path ###\ncontent`, joined by two newlines.
SKILL.md comes first. Directory supporting files are sorted by relative path;
ZIP supporting files retain archive order. A single ZIP wrapper directory is
removed when it contains SKILL.md. Hidden/unsupported files are skipped and listed;
`sample_meta.json` is excluded from model input to prevent source-label leakage.
Only supported text is classified, not skipped or binary content.

## Repository layout and publication

```text
app.py / parser.py / classifier.py / config.py   Desktop and inference pipeline
cli.py / benchmark.py / robustness.py           CLI and measurements
settings.example.json                          Public portable defaults
inference_contract.example.json                Public deployment prompt
examples/                                      Synthetic redistributable examples
samples/README.md                              Local sample import instructions
models/                                        Ignored model assets
samples/local/                                 Ignored imported research fixtures
reports/                                       Ignored generated results
```

Local settings, inference overrides, model assets, downloaded skills, reports,
research PDFs and environment caches are ignored by Git. Keep downloads outside
the source tree or in ignored `inbox/`. Do not explicitly force-add private files.
Application code is [MIT licensed](LICENSE); model/data rights remain separate:
[third-party materials](THIRD_PARTY.md). See [contribution guidance](CONTRIBUTING.md).
