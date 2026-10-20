"""Resumable public Hugging Face weight download using verified HTTP ranges."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def download(model_id, revision, filename, target, size, sha256, workers=64):
    target = Path(target)
    if target.exists() and target.stat().st_size == size:
        if digest(target) == sha256:
            return
    partial = target.with_suffix(target.suffix + '.partial')
    state = target.with_suffix(target.suffix + '.parts.json')
    chunk = 1024 * 1024
    done = set()
    if partial.exists() and state.exists():
        saved = json.loads(state.read_text())
        if saved.get('sha256') == sha256:
            previous_chunk = saved.get('chunk_size', 8 * 1024 * 1024)
            for index in saved['done']:
                done.update(range(index * previous_chunk // chunk, min((size + chunk - 1) // chunk, (index + 1) * previous_chunk // chunk)))
    elif not partial.exists():
        # Resume the interrupted snapshot_download's verified prefix.
        prefixes = sorted((target.parent / '.cache/huggingface/download').glob('*.incomplete'),
                          key=lambda p: p.stat().st_size, reverse=True)
        matching = [p for p in prefixes if sha256 in p.name]
        if matching:
            prefix = matching[0]
            complete_chunks = prefix.stat().st_size // chunk
            prefix.rename(partial)
            done = set(range(complete_chunks))
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    total = (size + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        for attempt in range(4):
            try:
                host = 'huggingface.co' if (index + attempt) % 2 == 0 else 'hf-mirror.com'
                url = f'https://{host}/{model_id}/resolve/{revision}/{filename}?download=true&part={start}'
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError(f'Invalid Content-Range for part {index}')
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError(f'Incomplete part {index}')
                offset = 0
                while offset < len(data):
                    written = os.pwrite(fd, data[offset:], start + offset)
                    if written <= 0:
                        raise OSError('Short write')
                    offset += written
                return index
            except Exception as exc:
                print(f'Part {index}, attempt {attempt+1}: {type(exc).__name__}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(total) if i not in done]
            failures = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    index = future.result()
                except Exception as exc:
                    failures.append(type(exc).__name__)
                    continue
                done.add(index)
                temp_state = state.with_suffix('.tmp')
                temp_state.write_text(json.dumps({'sha256': sha256, 'chunk_size': chunk, 'done': sorted(done)}))
                temp_state.replace(state)
                if len(done) % 50 == 0 or len(done) == total:
                    print(f'Weights: {len(done)}/{total} parts ({100*len(done)/total:.1f}%)', flush=True)
        os.fsync(fd)
        if failures:
            raise RuntimeError(f'{len(failures)} parts failed; progress saved. Rerun to resume.')
    finally:
        os.close(fd)
    if digest(partial) != sha256:
        raise RuntimeError('Weight SHA256 mismatch; refusing model file.')
    partial.replace(target)
    state.unlink(missing_ok=True)
    print(f'SHA256 verified: {sha256}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()
