"""Tkinter UI: all file/model work on one worker, all widgets on the Tk thread."""
import argparse
import json
from pathlib import Path
from queue import Queue, Empty
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from config import ROOT, Settings
from classifier import Classifier
from parser import parse_input


class App:
    def __init__(self, root, settings):
        self.root = root
        self.classifier = Classifier(settings)
        self.queue = Queue()
        self.busy = False
        self.closed = False
        self.last_result = None
        root.title('Agent Skill Triage · Local QLoRA')
        root.geometry('900x700')
        root.minsize(740, 570)
        root.protocol('WM_DELETE_WINDOW', self.close)
        frame = ttk.Frame(root, padding=22)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='Agent Skill Triage', font=('TkDefaultFont', 22, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Phân loại rủi ro trước khi cài skill · Xử lý cục bộ').pack(anchor='w', pady=(4, 16))
        self.path = tk.StringVar()
        row = ttk.Frame(frame)
        row.pack(fill='x')
        self.path_entry = ttk.Entry(row, textvariable=self.path)
        self.path_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.select_button = ttk.Button(row, text='Select File…', command=self.select)
        self.select_button.pack(side='right')
        self.folder_button = ttk.Button(row, text='Thư mục…', command=self.select_folder)
        self.folder_button.pack(side='right', padx=6)
        self.samples_button = ttk.Button(row, text='Samples…', command=lambda: self.select_folder(ROOT / 'samples/local'))
        self.samples_button.pack(side='right')
        self.serialized = tk.BooleanVar(value=False)
        self.mode = ttk.Checkbutton(frame, text='File văn bản đã serialization (full_text từ pipeline gốc)',
                                    variable=self.serialized)
        self.mode.pack(anchor='w', pady=10)
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=(0, 12))
        self.analyze_button = ttk.Button(controls, text='Analyze', command=self.analyze)
        self.analyze_button.pack(side='left')
        self.preview_button = ttk.Button(controls, text='Xem nội dung', command=lambda: self.analyze(preview=True))
        self.preview_button.pack(side='left', padx=(8, 0))
        self.clear_button = ttk.Button(controls, text='Clear', command=self.clear)
        self.clear_button.pack(side='left', padx=8)
        self.save_button = ttk.Button(controls, text='Lưu kết quả JSON', command=self.save, state='disabled')
        self.save_button.pack(side='left')
        self.spinner = ttk.Progressbar(controls, mode='indeterminate', length=150)
        self.spinner.pack(side='right')
        box = ttk.LabelFrame(frame, text='Kết quả', padding=14)
        box.pack(fill='x')
        self.result = tk.StringVar(value='Chưa phân tích')
        self.result_label = ttk.Label(box, textvariable=self.result, font=('TkDefaultFont', 21, 'bold'))
        self.result_label.pack(anchor='w')
        self.explanation = tk.StringVar(value='Chọn file/ZIP hoặc thư mục skill để đọc cả scripts và references.')
        ttk.Label(box, textvariable=self.explanation, wraplength=790).pack(anchor='w', pady=(8, 0))
        self.status = tk.StringVar(value='Sẵn sàng. Mô hình được nạp một lần khi phân tích lần đầu.')
        ttk.Label(frame, textvariable=self.status, wraplength=800).pack(anchor='w', pady=12)
        self.details = ScrolledText(frame, height=12, wrap='word', state='disabled')
        self.details.pack(fill='both', expand=True)
        ttk.Label(frame, text='HIGH_RISK là ưu tiên rà soát; NORMAL không bảo đảm gói an toàn.',
                  wraplength=800).pack(anchor='w', pady=(12, 0))
        self.root.after(80, self.poll)

    def select(self):
        path = filedialog.askopenfilename(initialdir=str(Path.home() / 'Downloads'), filetypes=[('Skill / ZIP / text', '*.md *.zip *.txt'), ('All files', '*')])
        if path:
            self.clear()
            self.path.set(path)

    def select_folder(self, initialdir=None):
        initialdir = initialdir or Path.home() / 'Downloads'
        path = filedialog.askdirectory(initialdir=str(initialdir), title='Chọn thư mục có SKILL.md ở gốc')
        if path:
            self.clear()
            self.path.set(path)

    def set_details(self, text):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', text)
        self.details.configure(state='disabled')

    def clear(self):
        self.last_result = None
        self.result.set('Chưa phân tích')
        self.result_label.configure(foreground='')
        self.explanation.set('Chọn đầu vào và nhấn Analyze.')
        self.status.set('Sẵn sàng.')
        self.set_details('')
        self.save_button.configure(state='disabled')

    def analyze(self, preview=False):
        if self.busy:
            return
        path = self.path.get().strip()
        if not path:
            messagebox.showinfo('Chọn đầu vào', 'Hãy chọn một file skill hoặc ZIP.')
            return
        serialized = self.serialized.get()
        self.clear()
        self.busy = True
        for widget in (self.select_button, self.folder_button, self.samples_button, self.preview_button, self.analyze_button, self.clear_button, self.mode, self.path_entry):
            widget.configure(state='disabled')
        self.spinner.start(12)
        self.status.set('Đang đọc đầu vào…')
        threading.Thread(target=self.work, args=(path, serialized, preview), daemon=True).start()

    def work(self, path, serialized, preview=False):
        start = time.perf_counter()
        try:
            package = parse_input(path, serialized=serialized)
            if preview:
                self.queue.put(('preview', package))
                return
            self.queue.put(('status', 'Đang nạp mô hình / phân tích trên máy…'))
            prediction = self.classifier.classify(package.text)
            result = dict(input=str(Path(path).resolve()), files=package.files, skipped=package.skipped,
                          **prediction.to_dict(), model_load_seconds=self.classifier.load_seconds,
                          total_seconds=time.perf_counter() - start)
            self.queue.put(('result', result))
        except Exception as exc:
            self.queue.put(('error', str(exc)))

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                    continue
                self.busy = False
                self.spinner.stop()
                for widget in (self.select_button, self.folder_button, self.samples_button, self.preview_button, self.analyze_button, self.clear_button, self.mode, self.path_entry):
                    widget.configure(state='normal')
                if kind == 'preview':
                    self.result.set('Đã đọc đầu vào')
                    self.explanation.set('Kiểm tra nội dung bên dưới rồi nhấn Analyze để phân loại.')
                    self.status.set(f'{len(value.files)} file văn bản · {len(value.skipped)} mục bỏ qua · chưa chạy model')
                    summary = json.dumps({'files': value.files, 'skipped': value.skipped}, ensure_ascii=False, indent=2)
                    self.set_details(summary + '\n\n' + value.text[:100000])
                    if len(value.text) > 100000:
                        self.details.configure(state='normal')
                        self.details.insert('end', '\n[Preview giới hạn 100.000 ký tự; inference vẫn dùng toàn bộ nội dung.]')
                        self.details.configure(state='disabled')
                elif kind == 'error':
                    self.result.set('INFERENCE_ERROR')
                    self.result_label.configure(foreground='#a32424')
                    self.explanation.set('Chưa có nhãn phân loại hợp lệ.')
                    self.status.set('Không hoàn tất. Xem chi tiết bên dưới.')
                    self.set_details(value)
                else:
                    self.last_result = value
                    label = value['label']
                    self.result.set(label)
                    self.result_label.configure(foreground='#a32424' if label == 'HIGH_RISK' else '#176b45')
                    self.explanation.set('Ưu tiên rà soát trước khi cài đặt.' if label == 'HIGH_RISK'
                                         else 'Mô hình không gắn cờ rủi ro cao cho đầu vào này.')
                    self.status.set(f"Hoàn tất · {value['package_tokens']:,} token · inference {value['inference_seconds']:.3f}s")
                    self.set_details(json.dumps(value, ensure_ascii=False, indent=2))
                    self.save_button.configure(state='normal')
        except Empty:
            pass
        self.root.after(80, self.poll)

    def save(self):
        if self.last_result is None:
            return
        reports = ROOT / 'reports'
        reports.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=str(reports), initialfile=datetime.now().strftime('triage-%Y%m%d-%H%M%S.json'), defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                Path(path).write_text(json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding='utf-8')
            except OSError as exc:
                messagebox.showerror('Không lưu được', str(exc))

    def close(self):
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', type=Path, help='ZIP, file văn bản hoặc thư mục skill tải về')
    parser.add_argument('--analyze', action='store_true', help='Chạy ngay trên đường dẫn đầu vào')
    parser.add_argument('--settings', type=Path, default=ROOT / 'settings.json')
    args = parser.parse_args()
    root = tk.Tk()
    try:
        app = App(root, Settings.read(args.settings))
        if args.input:
            app.path.set(str(args.input.expanduser().resolve()))
            if args.analyze:
                root.after(200, app.analyze)
    except Exception as exc:
        messagebox.showerror('Lỗi cấu hình', str(exc))
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
