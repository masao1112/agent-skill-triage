"""One-time download; never imported by the offline desktop application."""
import argparse
import json
from config import ROOT


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='main', help='Use the original training base revision when available.')
    args = parser.parse_args()
    model_id = 'Qwen/Qwen3-0.6B'
    info = HfApi().model_info(model_id, revision=args.revision, files_metadata=True)
    sha = info.sha
    target = ROOT / 'models/base'
    snapshot_download(model_id, revision=sha, local_dir=target,
                      allow_patterns=['*.json', '*.jinja', 'merges.txt', 'vocab.json', 'README.md', 'LICENSE'])
    from download_weights import download
    weight = next(f for f in info.siblings if f.rfilename == 'model.safetensors')
    download(model_id, sha, weight.rfilename, target / weight.rfilename, weight.size, weight.lfs.sha256)
    (ROOT / 'models/base-provenance.json').write_text(json.dumps(
        {'model_id': model_id, 'revision': sha, 'requested_revision': args.revision}, indent=2) + '\n')
    print(f'Base model ready: {target}; revision={sha}')


if __name__ == '__main__':
    main()
