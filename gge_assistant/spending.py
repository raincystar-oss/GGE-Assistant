"""User spending policy; historical entries are never reset or rewritten."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from .artifacts import atomic_write_json
from .lock import AppLock

NANO = 1_000_000_000


class SpendingError(ValueError):
    pass


def money(value):
    try:
        amount = Decimal(str(value).strip().replace(',', '.'))
        if not amount.is_finite() or amount < 0 or amount > 1_000_000 or amount.as_tuple().exponent < -8:
            raise ValueError()
        return int(amount * NANO)
    except (ValueError, InvalidOperation):
        raise SpendingError('Сумма должна быть неотрицательным числом в долларах, например 0,10.') from None


def cost(entry):
    return entry.get('cost_nano', entry['reserved_nano'])


def fingerprint(entries):
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate(data):
    policy = data.get('ui_budget')
    if policy is None:
        return
    start = policy.get('start_entry')
    if (policy.get('schema') != 1 or type(start) is not int or not 0 <= start <= len(data['entries'])
            or type(policy.get('enabled')) is not bool
            or any(type(policy.get(k)) is not int or policy[k] < 0
                   for k in ('limit_nano', 'max_calls', 'min_balance_nano'))
            or policy.get('previous_entries_sha256') != fingerprint(data['entries'][:start])):
        raise SpendingError('Настройки расходов или сохранённая история повреждены. Отправка остановлена.')


def summary(data):
    validate(data)
    p = data.get('ui_budget')
    entries = data['entries']
    start = p['start_entry'] if p else len(entries)
    current = entries[start:]
    used = sum(cost(e) for e in current)
    limit = p['limit_nano'] if p else 0
    calls = p['max_calls'] if p else 0
    return dict(configured=p is not None, enabled=bool(p and p['enabled']),
        limit_nano=limit, max_calls=calls, used_nano=used, used_calls=len(current),
        remaining_nano=max(0, limit - used), remaining_calls=max(0, calls - len(current)),
        historical_nano=sum(cost(e) for e in entries[:start]), historical_calls=start,
        uncertain_nano=sum(cost(e) for e in entries if e['status'] != 'complete'),
        uncertain_calls=sum(e['status'] != 'complete' for e in entries),
        blocking_calls=sum(e['status'] != 'complete' for e in current),
        min_balance_nano=p['min_balance_nano'] if p else (3_500_000_000 if entries else 0))


def check(data, reserve):
    state = summary(data)
    if not state['enabled']:
        raise SpendingError('Платные ответы отключены. Откройте «Настройки расходов» и сохраните лимит.')
    if state['blocking_calls']:
        raise SpendingError('Предыдущий запрос имеет неопределённую стоимость. Резерв сохранён; новые запросы заблокированы до проверки.')
    if state['remaining_calls'] < 1 or reserve > state['remaining_nano']:
        raise SpendingError('Лимит расходов или числа запросов исчерпан. Проверьте «Настройки расходов».')
    return state


def read_ledger(path):
    # Imported lazily to avoid a dependency cycle with the budget client's guard.
    from .rag_budget import BudgetLedger
    if path.exists():
        raw = json.loads(path.read_text(encoding='utf-8'))
        return BudgetLedger(path, raw.get('account_id')).read()
    return BudgetLedger(path, None).read()


def status(path):
    return summary(read_ledger(Path(path)))


def configure(path, amount, calls, enabled, min_balance):
    path = Path(path)
    limit, floor = money(amount or '0'), money(min_balance or '0')
    if isinstance(calls, bool) or not str(calls).strip().isascii() or not str(calls).strip().isdigit():
        raise SpendingError('Число запросов должно быть целым неотрицательным числом.')
    count = int(str(calls).strip())
    if count > 1_000_000 or type(enabled) is not bool or enabled and (limit <= 0 or count <= 0):
        raise SpendingError('Для включения платных ответов укажите положительную сумму и число запросов.')
    with AppLock(path.with_suffix('.lock')):
        data = read_ledger(path)
        policy = data.get('ui_budget')
        if policy is None:
            policy = dict(schema=1, start_entry=len(data['entries']),
                          previous_entries_sha256=fingerprint(data['entries']), changes=[])
        change = dict(utc=datetime.now(timezone.utc).isoformat(), enabled=enabled,
                      limit_nano=limit, max_calls=count, min_balance_nano=floor)
        policy = {**policy, **change, 'changes': [*policy['changes'], change]}
        data['ui_budget'] = policy
        atomic_write_json(path, data)
        return summary(data)
