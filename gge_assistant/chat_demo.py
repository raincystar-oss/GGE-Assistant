"""Local chat and material import interface."""
import argparse
import json
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import scrolledtext

from .rag_context import safe_text
from .source_links import SourceLinks
from .import_dialog import AddMaterialDialog
from .expense_dialog import ExpenseDialog
from . import spending
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / 'data' / 'rag_pilot' / 'queries'


def failure_message(stderr):
    lowered = stderr.casefold()
    if 'настройки расходов' in lowered or 'неопределённую стоимость' in lowered:
        return safe_text(stderr.removeprefix('STOP:').strip())
    if 'budget' in lowered or 'call limit' in lowered or 'grant limit' in lowered:
        return ('Ответ не сформирован: разрешённый лимит платных запросов исчерпан. '
                'Для нового ответа нужен дополнительный бюджет. '
                'Сохранённые ответы доступны только в той базе, для которой они были получены.')
    if 'uncertain cost' in lowered:
        return 'Ответ не сформирован: предыдущий платный запрос требует проверки учёта расходов.'
    return ('Запрос не выполнен. Проверьте, что AnythingLLM запущен и база доступна.\n'
            + safe_text(stderr))


def install_edit_shortcuts(widget):
    # Windows virtual key codes remain stable when the keyboard layout changes.
    def control_key(event):
        action = {86: '<<Paste>>', 67: '<<Copy>>', 88: '<<Cut>>'}.get(event.keycode)
        if action:
            widget.event_generate(action)
            return 'break'
        if event.keycode == 65:
            widget.tag_add('sel', '1.0', 'end-1c')
            return 'break'

    widget.bind('<Control-KeyPress>', control_key)
    menu = tk.Menu(widget, tearoff=False)
    for label, action in (('Вырезать', '<<Cut>>'), ('Копировать', '<<Copy>>'), ('Вставить', '<<Paste>>')):
        menu.add_command(label=label, command=lambda event=action: widget.event_generate(event))

    def popup(event):
        widget.focus_set()
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return 'break'
    widget.bind('<Button-3>', popup)


def saved_answers(workspace=None):
    results = {}
    status_path = ROOT / 'data' / 'rag_pilot' / 'index_status.json'
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        status = {}
    for path in QUERIES.glob('*.json'):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            if workspace and item.get('corpus', {}).get('workspace') != workspace:
                continue
            if (workspace and status.get('workspace') == workspace
                    and item.get('corpus', {}).get('corpus_fingerprint') != status.get('corpus_fingerprint')):
                continue
            answer = item.get('answer') or {}
            if (item.get('question') and answer.get('text') and not answer.get('unknown_citations')
                    and not answer.get('condition_warnings') and not answer.get('consistency_warnings')
                    and answer.get('finish_reason') != 'length'):
                results[item['question']] = answer
        except (OSError, ValueError, TypeError):
            continue
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='GGE Assistant chat')
    parser.add_argument('--config', help='Project configuration for the selected corpus')
    args = parser.parse_args(argv)
    links = SourceLinks(args.config)
    workspace = links.app.anythingllm_workspace_slug
    saved = saved_answers(workspace)
    root = tk.Tk()
    root.title('GGE Assistant')
    root.geometry('900x700')
    root.minsize(600, 420)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(2, weight=1)
    tk.Label(root, text='GGE Assistant', font=('Segoe UI', 18, 'bold')).grid(row=0, column=0, pady=(15, 3))
    tk.Label(root, text=f'Вопросы по материалам вебинаров · {workspace}', font=('Segoe UI', 10)).grid(row=1, column=0)
    chat = scrolledtext.ScrolledText(root, height=8, wrap='word', font=('Segoe UI', 11), state='disabled')
    chat.grid(row=2, column=0, sticky='nsew', padx=18, pady=12)
    entry = tk.Text(root, height=4, wrap='word', font=('Segoe UI', 11))
    install_edit_shortcuts(entry)
    entry.grid(row=3, column=0, sticky='ew', padx=18, pady=(0, 8))
    buttons = tk.Frame(root)
    buttons.grid(row=4, column=0, sticky='ew', padx=18, pady=(0, 15))

    chat.tag_configure('user', foreground='#2359a6', font=('Segoe UI', 11, 'bold'))
    chat.tag_configure('assistant', foreground='#166447', font=('Segoe UI', 11, 'bold'))
    chat.tag_configure('link', foreground='#2359a6', underline=True)
    link_number = 0
    importing = False
    dialog = None
    expenses = None
    ledger_path = ROOT / 'data' / 'rag_pilot' / 'budget.json'

    def open_expenses():
        nonlocal expenses
        if expenses and expenses.window.winfo_exists():
            expenses.refresh(fill=True)
            expenses.window.lift()
            return
        expenses = ExpenseDialog(root, ledger_path, install_edit_shortcuts)

    def set_importing(value):
        nonlocal importing
        importing = value
        send.configure(state='disabled' if value else 'normal')
        add.configure(state='disabled' if value else 'normal')
        expense_button.configure(state='disabled' if value else 'normal')

    def import_complete(result):
        nonlocal links, saved
        links = SourceLinks(args.config)
        saved = saved_answers(workspace)
        show('Материал добавлен. Можно задать вопрос по обновлённой базе.')

    def open_import():
        nonlocal dialog
        if dialog and dialog.window.winfo_exists():
            dialog.window.lift()
            return
        dialog = AddMaterialDialog(root, args.config, install_edit_shortcuts, set_importing, import_complete, workspace=workspace)

    def close_chat():
        if importing:
            messagebox.showinfo('Обработка материала', 'Дождитесь завершения добавления материала.', parent=root)
        else:
            root.destroy()
    root.protocol('WM_DELETE_WINDOW', close_chat)

    def show(text, tag=None):
        chat.configure(state='normal')
        chat.insert('end', text + '\n\n', tag)
        chat.see('end')
        chat.configure(state='disabled')

    def display_answer(answer):
        nonlocal link_number
        show('Ассистент:\n' + safe_text(answer['text']), 'assistant')
        show('Источники: обозначения [S1], [S2] в ответе соответствуют материалам ниже.')
        for group in links.groups(answer.get('sources', [])):
            def timecode(value):
                seconds = max(0, int(float(value or 0)))
                hours, remainder = divmod(seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f'{hours}:{minutes:02}:{seconds:02}' if hours else f'{minutes}:{seconds:02}'
            fragments = '\n'.join(
                f"[{f['source_id']}] {timecode(f['start'])}–{timecode(f['end'])}"
                for f in group['fragments'])
            show(safe_text(str(group['title'])) + '\nДата: ' + str(group['date'] or 'не указана') + '\n' + fragments)
            url = group['url']
            if url:
                link_number += 1
                tag = f'link_{link_number}'
                chat.configure(state='normal')
                chat.insert('end', 'Открыть страницу вебинара\n\n', ('link', tag))
                chat.tag_bind(tag, '<Button-1>', lambda event, target=url: webbrowser.open(target))
                chat.tag_bind(tag, '<Enter>', lambda event: chat.configure(cursor='hand2'))
                chat.tag_bind(tag, '<Leave>', lambda event: chat.configure(cursor='xterm'))
                chat.configure(state='disabled')
            else:
                show('Ссылка на страницу материала пока не указана.')
        chat.see('end')

    def submit():
        if importing or str(send['state']) == 'disabled':
            return
        question = entry.get('1.0', 'end').strip()
        if not question:
            return
        if question in saved:
            entry.delete('1.0', 'end')
            show('Вы: ' + question, 'user')
            display_answer(saved[question])
            return
        try:
            settings = spending.status(ledger_path)
            if not settings['enabled']:
                show('Платные ответы отключены. Задайте лимит в «Настройках расходов». Вопрос оставлен в поле ввода.')
                open_expenses()
                return
        except Exception:
            show('Не удалось прочитать настройки расходов. Отправка остановлена до проверки журнала.')
            return
        entry.delete('1.0', 'end')
        show('Вы: ' + question, 'user')
        send.configure(state='disabled')
        add.configure(state='disabled')
        expense_button.configure(state='disabled')
        show('Ищу ответ в материалах…', 'assistant')
        def work():
            try:
                command = [sys.executable, '-m', 'gge_assistant.rag_pilot']
                if args.config:
                    command.extend(['--config', args.config])
                command.extend(['ask', question, '--answer'])
                result = subprocess.run(command,
                                        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=720)
                text = ('Ассистент:\n' + result.stdout if result.returncode == 0
                        else failure_message(result.stderr))
            except (OSError, subprocess.TimeoutExpired):
                text = 'Поиск не выполнен. Проверьте запуск AnythingLLM и повторите попытку.'
            # Use the same structured rendering for newly generated and cached answers.
            fresh = saved_answers(workspace).get(question) if 'result' in locals() and result.returncode == 0 else None
            def finish():
                if fresh:
                    saved[question] = fresh
                    display_answer(fresh)
                else:
                    show(safe_text(text))
                send.configure(state='normal')
                add.configure(state='normal')
                expense_button.configure(state='normal')
            root.after(0, finish)
        threading.Thread(target=work, daemon=True).start()

    send = tk.Button(buttons, text='Отправить', command=submit, padx=20)
    send.pack(side='right')
    add = tk.Button(buttons, text='Добавить материал', command=open_import, padx=15)
    add.pack(side='left')
    expense_button = tk.Button(buttons, text='Настройки расходов', command=open_expenses, padx=10)
    expense_button.pack(side='left', padx=(8, 0))
    root.update_idletasks()
    root.minsize(max(600, buttons.winfo_reqwidth() + 36), 420)
    show('Введите вопрос и нажмите «Отправить». Ответ формируется по материалам вебинаров; обозначения [S1], [S2] указывают на источники под ответом.')
    root.mainloop()


if __name__ == '__main__':
    main()

