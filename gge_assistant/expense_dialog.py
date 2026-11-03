"""Local expense settings; opening/saving never sends a provider request."""
import tkinter as tk

from . import spending
from .rag_budget import usd


class ExpenseDialog:
    def __init__(self, parent, ledger_path, shortcuts, on_saved=lambda: None):
        self.path, self.on_saved = ledger_path, on_saved
        self.window = tk.Toplevel(parent)
        self.window.title('Настройки расходов — GGE Assistant')
        self.window.transient(parent)
        self.window.grab_set()
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar()
        tk.Label(self.window, text='Лимит на дальнейшую работу', font=('Segoe UI', 13, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=18, pady=(15, 8))
        tk.Label(self.window, text='Счётчики начинаются при первой настройке этого окна. '
                 'Изменение лимита их не обнуляет. Настройки общие для всех баз. '
                 'Стоимость учитывается по верхней оценке, включая резервы.',
                 wraplength=580, justify='left').grid(row=1, column=0, columnspan=2, sticky='ew', padx=18)
        self.fields = []
        for row, label in enumerate(('Максимальная сумма, USD', 'Максимальное число платных запросов',
                                     'Минимальный остаток на счёте DeepSeek, USD'), 2):
            tk.Label(self.window, text=label, anchor='w').grid(row=row, column=0, sticky='w', padx=18, pady=6)
            field = tk.Text(self.window, height=1, width=16, font=('Segoe UI', 10))
            field.grid(row=row, column=1, sticky='ew', padx=(5, 18), pady=6)
            shortcuts(field)
            self.fields.append(field)
        tk.Checkbutton(self.window, text='Включить платные ответы', variable=self.enabled).grid(
            row=5, column=0, columnspan=2, sticky='w', padx=18, pady=8)
        self.metrics = tk.StringVar()
        tk.Label(self.window, textvariable=self.metrics, justify='left', anchor='w', wraplength=580).grid(
            row=6, column=0, columnspan=2, sticky='ew', padx=18, pady=8)
        tk.Label(self.window, textvariable=self.status, wraplength=580, justify='left').grid(
            row=7, column=0, columnspan=2, sticky='ew', padx=18, pady=8)
        tk.Button(self.window, text='Сохранить настройки', command=self.save, padx=15).grid(
            row=8, column=0, columnspan=2, sticky='e', padx=18, pady=(4, 15))
        self.refresh(fill=True)
        self.window.update_idletasks()
        self.window.minsize(self.window.winfo_reqwidth(), self.window.winfo_reqheight())

    def refresh(self, fill=False):
        try:
            state = spending.status(self.path)
        except Exception:
            self.status.set('Не удалось прочитать журнал расходов. Сохранение заблокировано до проверки файла.')
            return
        if fill:
            self.enabled.set(state['enabled'])
            for field, value in zip(self.fields, (usd(state['limit_nano']), str(state['max_calls']), usd(state['min_balance_nano']))):
                field.delete('1.0', 'end')
                field.insert('1.0', value)
        self.metrics.set(
            f"Текущий лимит: ${usd(state['limit_nano'])} / {state['max_calls']} запросов\n"
            f"Учтено с первой настройки: ${usd(state['used_nano'])} / {state['used_calls']} запросов\n"
            f"Осталось: ${usd(state['remaining_nano'])} / {state['remaining_calls']} запросов\n\n"
            f"Прежняя история сохранена отдельно: ${usd(state['historical_nano'])} / {state['historical_calls']} запросов\n"
            f"Неподтверждённые попытки во всей истории: {state['uncertain_calls']} "
            f"(учтено по резерву ${usd(state['uncertain_nano'])})")
        if state['blocking_calls']:
            self.status.set('Есть незавершённый новый запрос. Его резерв сохранён; увеличение лимита не снимает блокировку.')
        else:
            self.status.set('Платные ответы включены.' if state['enabled'] else 'Платные ответы отключены. Поиск и добавление материалов доступны.')

    def save(self):
        try:
            amount, calls, floor = [f.get('1.0', 'end').strip() for f in self.fields]
            spending.configure(self.path, amount, calls, self.enabled.get(), floor)
        except spending.SpendingError as exc:
            self.status.set(str(exc))
            return
        except Exception:
            self.status.set('Не удалось сохранить настройки. Проверьте журнал расходов; история не сбрасывается.')
            return
        self.refresh()
        self.on_saved()
