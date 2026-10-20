# Contributing

Use Python 3.12. Keep parsing independent of model libraries so input and UI tests
run with the standard library. On Linux, install Tkinter and provide a display
(or Xvfb) for the GUI tests; without a display those tests are skipped.

```bash
python -m unittest discover -s tests -v
```

Changes to serialization, token limits, prompts, decoding or label mapping alter
inference behavior. Document them and use a fixed, independent validation set.
Do not tune on held-out test samples or report synthetic examples as test accuracy.

Never execute input package scripts or import fixtures as assistant skills.
Test directory links, ZIP traversal/links/bombs, size limits, encoding errors and
invalid model output when changing input handling. Use temporary directories.
Do not commit private skills, generated reports, credentials or model checkpoints.
Keep secrets and local paths out of bug reports; provide a minimal synthetic input.
