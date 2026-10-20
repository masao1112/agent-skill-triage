# Local sample packages

`samples/local/` contains user-imported research fixtures and is ignored by Git.
The original directories are preserved. The importer copies complete packages,
records SHA256/provenance, and creates a benchmark manifest with source labels.
Labels inherited from folder names are corpus annotations, not new expert verdicts.

```bash
.venv/bin/python import_samples.py /path/to/raw_samples --per-group 1 --validate-tokens
.venv/bin/python cli.py benchmark samples/local/manifest.json --output reports/local-samples.json
```

Groups: G1_normal, G2_prompt_injection, G3_data_exfiltration, G4_jailbreak.
Use the **Samples…** button and select an individual folder containing SKILL.md.
`sample_meta.json` is omitted from model input to avoid leaking the source label.
The complete original files remain available locally for inspection.

Imported packages may contain adversarial text or scripts. The app reads data;
it does not run scripts, install packages, or register these as assistant skills.
Do not redistribute third-party fixtures without checking their licenses.
Synthetic, redistributable app examples are under `examples/`.
