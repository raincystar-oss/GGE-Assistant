"""Tk form; all heavy work runs outside the UI thread."""
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .material_import import PHASES, ImportProblem, import_material, validate_input


class AddMaterialDialog:
    def __init__(self, parent, config, shortcuts, on_busy, on_complete, workspace=''):
        self.window = tk.Toplevel(parent)
        self.window.title('Добавить материал — ' + (workspace or 'GGE Assistant'))
        self.window.geometry('650x560')
        self.window.minsize(540, 500)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.columnconfigure(0, weight=1)
        self.running = False
        self.on_busy, self.on_complete = on_busy, on_complete
        self.config = config
        self.events = queue.Queue()
        self.fields = []
        for row, (label, height) in enumerate((('Ссылка на запись RuTube', 2),
                ('Название материала', 1), ('Дата вебинара (ГГГГ-ММ-ДД)', 1),
                ('Страница мероприятия ГГЭ (необязательно)', 2))):
            tk.Label(self.window, text=label, anchor='w').grid(row=row * 2, column=0, sticky='ew', padx=18, pady=(10, 2))
            field = tk.Text(self.window, height=height, wrap='word', font=('Segoe UI', 10))
            field.grid(row=row * 2 + 1, column=0, sticky='ew', padx=18)
            shortcuts(field)
            self.fields.append(field)
        self.status = tk.StringVar(value='Обработка выполняется локально и может занять длительное время. Ответ DeepSeek не запрашивается.')
        tk.Label(self.window, textvariable=self.status, wraplength=590, justify='left').grid(row=8, column=0, sticky='ew', padx=18, pady=12)
        self.bar = ttk.Progressbar(self.window, mode='indeterminate')
        self.bar.grid(row=9, column=0, sticky='ew', padx=18)
        self.window.rowconfigure(10, weight=1)
        self.start = tk.Button(self.window, text='Обработать', command=self.submit, padx=18)
        self.start.grid(row=11, column=0, sticky='e', padx=18, pady=15)
        self.window.protocol('WM_DELETE_WINDOW', self.close)

    def close(self):
        if self.running:
            messagebox.showinfo('Обработка материала', 'Дождитесь завершения обработки перед закрытием окна.', parent=self.window)
        else:
            self.window.destroy()

    def submit(self):
        if self.running:
            return
        try:
            item = validate_input(*(field.get('1.0', 'end').strip() for field in self.fields))
        except ImportProblem as exc:
            self.status.set(str(exc))
            return
        self.running = True
        self.start.configure(state='disabled')
        for field in self.fields:
            field.configure(state='disabled')
        self.on_busy(True)
        self.bar.start(15)
        self.status.set(PHASES['checking'])

        def work():
            try:
                result = import_material(item, self.config, lambda stage: self.events.put(('progress', stage)))
                self.events.put(('done', result))
            except ImportProblem as exc:
                self.events.put(('error', str(exc)))
            except Exception:
                self.events.put(('error', 'Не удалось завершить обработку. Требуется проверка локальных настроек.'))
        threading.Thread(target=work, daemon=True).start()
        self.window.after(100, self.poll)

    def poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                self.status.set(PHASES.get(value, 'Выполняется обработка…'))
            else:
                self.running = False
                self.bar.stop()
                self.on_busy(False)
                if kind == 'done':
                    self.status.set('Материал доступен для поиска.' + (
                        ' Неустойчивые фрагменты исключены из расшифровки; сохранён отчёт проверки.'
                        if value.get('partial') else ''))
                    self.start.configure(text='Закрыть', state='normal', command=self.close)
                    self.on_complete(value)
                else:
                    self.status.set(value)
                    self.start.configure(state='normal')
                    for field in self.fields:
                        field.configure(state='normal')
                return
        self.window.after(100, self.poll)
