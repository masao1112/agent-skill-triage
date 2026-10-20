"""Read untrusted packages as data, in memory; never extract or execute them."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
import os
import zipfile
from config import Limits


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class Package:
    text: str
    files: tuple[str, ...]
    skipped: tuple[str, ...] = ()


TEXT_SUFFIXES = frozenset(('md txt py sh bash zsh js mjs cjs ts tsx jsx json yaml yml '
                           'toml ini cfg conf csv xml html css sql r ps1 bat cmd '
                           'rs go java c h cpp hpp rb php lua dockerfile jinja jinja2').split())


def safe_name(name):
    if not name or '\\' in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise InputError(f'Đường dẫn ZIP không hợp lệ: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '###' in name:
        raise InputError(f'Đường dẫn ZIP nguy hiểm: {name!r}')
    return p.as_posix()


def decode_text(data, name):
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InputError(f'{name}: encoding không phải UTF-8.') from exc
    if '\x00' in text:
        raise InputError(f'{name}: nội dung nhị phân/NUL không được hỗ trợ.')
    return text


def serialize(files):
    # Exact delimiter and separator from build_full_text in the source notebook.
    if not files or not any(text.strip() for _, text in files):
        raise InputError('Không có nội dung văn bản hợp lệ.')
    return '\n\n'.join(f'### FILE: {name} ###\n{text}' for name, text in files)


def parse_input(path, *, serialized=False, limits=Limits()):
    path = Path(path)
    if path.is_symlink():
        raise InputError('Không hỗ trợ symlink đầu vào.')
    if path.is_dir():
        if serialized:
            raise InputError('Chế độ full_text chỉ áp dụng file văn bản.')
        return read_directory(path, limits)
    if not path.is_file():
        raise InputError('File không tồn tại hoặc không phải file thường.')
    if path.stat().st_size > limits.max_input_bytes:
        raise InputError('File đầu vào vượt giới hạn kích thước.')
    try:
        if path.suffix.lower() == '.zip':
            if serialized:
                raise InputError('Chế độ văn bản đã serialization chỉ áp dụng file văn bản.')
            return read_zip(path, limits)
        if path.stat().st_size > limits.max_file_bytes:
            raise InputError('File văn bản vượt giới hạn kích thước.')
        with path.open('rb') as stream:
            data = stream.read(limits.max_file_bytes + 1)
        if len(data) > limits.max_file_bytes:
            raise InputError('File văn bản vượt giới hạn kích thước.')
        text = decode_text(data, path.name)
        if not text.strip():
            raise InputError('File rỗng.')
        if serialized:
            if not text.startswith('### FILE: ') or ' ###\n' not in text:
                raise InputError('Thiếu định dạng ### FILE: đường/dẫn ### trong full_text.')
            return Package(text, (path.name,))
        name = safe_name(path.name)
        return Package(serialize([(name, text)]), (name,))
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
        raise InputError(f'Không đọc được đầu vào: {exc}') from exc


def read_zip(path, limits):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > limits.max_files:
            raise InputError('ZIP có quá nhiều mục.')
        total = 0
        seen = set()
        candidates = []
        for info in entries:
            name = safe_name(info.filename)
            if name in seen:
                raise InputError(f'ZIP có đường dẫn trùng: {name}')
            seen.add(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise InputError('ZIP chứa symlink hoặc file đặc biệt.')
            if info.flag_bits & 1:
                raise InputError('ZIP mã hóa không được hỗ trợ.')
            total += info.file_size
            if info.file_size > limits.max_file_bytes or total > limits.max_total_bytes:
                raise InputError('Nội dung giải nén vượt giới hạn kích thước.')
            if info.file_size / max(1, info.compress_size) > limits.max_ratio:
                raise InputError('Tỷ lệ nén ZIP quá cao.')
            if not info.is_dir():
                candidates.append((name, info))
        # Strip exactly one enclosing directory, only if it contains SKILL.md.
        names = [name for name, _ in candidates]
        wrapper = ''
        if names and all('/' in n and n.split('/')[0] == names[0].split('/')[0] for n in names):
            prefix = names[0].split('/')[0] + '/'
            if prefix + 'SKILL.md' in names:
                wrapper = prefix
        files, skipped = [], []
        for name, info in candidates:
            name = name[len(wrapper):]
            if name == 'sample_meta.json' or any(p.startswith('.') for p in PurePosixPath(name).parts):
                skipped.append(name)
                continue
            suffix = PurePosixPath(name).suffix.lstrip('.').lower()
            if suffix not in TEXT_SUFFIXES and PurePosixPath(name).name.lower() not in ('dockerfile', 'makefile'):
                skipped.append(name)
                continue
            with archive.open(info) as stream:
                data = stream.read(limits.max_file_bytes + 1)
            if len(data) > limits.max_file_bytes:
                raise InputError('File ZIP vượt giới hạn khi đọc.')
            files.append((name, decode_text(data, name)))
        if not any(name == 'SKILL.md' for name, _ in files):
            raise InputError('Gói ZIP phải có một SKILL.md ở gốc hoặc trong một thư mục bao ngoài.')
        # SKILL.md first, other entries retain ZIP manifest order (no invented sorting).
        files = [f for f in files if f[0] == 'SKILL.md'] + [f for f in files if f[0] != 'SKILL.md']
        return Package(serialize(files), tuple(n for n, _ in files), tuple(skipped))


def read_directory(path, limits=Limits()):
    """Read a downloaded skill folder, never following links or executing files."""
    path = Path(path)
    files, skipped = [], []
    count = total = 0
    try:
        if not (path / 'SKILL.md').is_file():
            raise InputError('Thư mục phải chứa SKILL.md ở gốc.')
        def walk_error(error):
            raise error
        for parent, dirs, names in os.walk(path, followlinks=False, onerror=walk_error):
            count += len(dirs) + len(names)
            if count > limits.max_files:
                raise InputError('Thư mục có quá nhiều mục.')
            for name in dirs + names:
                item = Path(parent) / name
                if item.is_symlink():
                    raise InputError('Thư mục chứa symlink; không đọc đường dẫn liên kết.')
            hidden = [name for name in dirs if name.startswith('.')]
            skipped.extend((Path(parent) / name).relative_to(path).as_posix() + '/' for name in hidden)
            dirs[:] = sorted(name for name in dirs if not name.startswith('.'))
            for name in sorted(names):
                item = Path(parent) / name
                relative = safe_name(item.relative_to(path).as_posix())
                mode = item.stat().st_mode
                if not stat.S_ISREG(mode):
                    raise InputError('Thư mục chứa file đặc biệt.')
                size = item.stat().st_size
                total += size
                if size > limits.max_file_bytes or total > limits.max_total_bytes:
                    raise InputError('Nội dung thư mục vượt giới hạn kích thước.')
                suffix = item.suffix.lstrip('.').lower()
                if (relative == 'sample_meta.json' or name.startswith('.') or
                    (suffix not in TEXT_SUFFIXES and name.lower() not in ('dockerfile', 'makefile'))):
                    skipped.append(relative)
                    continue
                # O_NOFOLLOW also guards a final-component link swap on Linux.
                descriptor = os.open(item, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                with os.fdopen(descriptor, 'rb') as stream:
                    data = stream.read(limits.max_file_bytes + 1)
                if len(data) > limits.max_file_bytes:
                    raise InputError('File vượt giới hạn khi đọc.')
                files.append((relative, decode_text(data, relative)))
        files.sort(key=lambda item: (item[0] != 'SKILL.md', item[0]))
        return Package(serialize(files), tuple(name for name, _ in files), tuple(skipped))
    except OSError as exc:
        raise InputError(f'Không đọc được thư mục: {exc}') from exc
