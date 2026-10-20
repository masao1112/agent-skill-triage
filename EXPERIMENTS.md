# Deployment evaluation protocol

Use a fixed checkpoint, tokenizer, serialization, prompt, label mapping and runtime
precision throughout a comparison. Record their hashes and the base revision.
Never update model weights or select prompts using held-out test examples.

## Functional correctness

Create a fixed manifest with full packages or original serialized text. Obtain
`reference_label` independently from the original inference pipeline, then compare
labels and valid output rates. To establish serialization equivalence, compare
serialized bytes and tokenizer IDs as well. Missing references remain null.
The default deployment prompt is not verified against the training prompt.

## Latency

Run `cli.py benchmark manifest.json --warmup 1 --output reports/benchmark.json`.
Report n, input token lengths, mean, median and nearest-rank p95. Model load is
separate from each sample; generation, tokenization and parsing are separately
accounted for. CLI imports torch before model-load timing; GUI initial load may
include more runtime imports. Do not extrapolate short-demo latency to 10k tokens.

## Resources

Use a fresh process and report CPU, GPU, physical RAM, OS, package versions and
actual inference dtype/quantization. Process RSS peak is sampled every 10ms over
model loading, warmup and inference; short peaks can be missed. Report GPU peak
allocated/reserved memory when applicable and installation file sizes. Model
hashing takes place after resource sampling. Filesystem caching can affect load.

## Input robustness

Run `python -m unittest discover -s tests -v` and, with a saved tokenizer,
`python robustness.py`. Tests cover malformed/empty input, bad UTF-8, ZIP traversal,
links/duplicate entries/compression limits, directory links/special files/size
limits, token overflow, invalid output and UI recovery. These tests do not measure
malicious-skill detection accuracy. Keep generated reports local under reports/.

## Local corpus samples

`import_samples.py` preserves original packages and records source labels and
checksums in ignored `samples/local/`. Labels from corpus groups are annotations,
not independently established maliciousness. sample_meta.json is excluded from
model input to prevent label leakage. Directory supporting files are sorted by
relative path; ZIP order follows the archive. Report these choices when comparing
against a different preprocessing pipeline. Reusing research samples does not
make them a new, independent held-out benchmark.
