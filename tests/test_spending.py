import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from gge_assistant.rag_budget import BudgetLedger, BudgetedDeepSeek, BudgetError
from gge_assistant import spending
from gge_assistant import rag_pilot
from unittest.mock import patch

MESSAGES = [{'role': 'system', 'content': 'Sources only'}, {'role': 'user', 'content': 'Test question'}]


class SpendingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'budget.json'
        self.ledger = BudgetLedger(self.path, 'test')

    def configure(self, amount='0.10', calls='3', enabled=True, floor='0'):
        return spending.configure(self.path, amount, calls, enabled, floor)

    def test_no_settings_means_no_requests_or_network(self):
        def handler(request):
            self.fail('No HTTP call is allowed before configuration')
        with httpx.Client(base_url='https://api.deepseek.com', transport=httpx.MockTransport(handler)) as client:
            paid = BudgetedDeepSeek(self.path, client=client, key='test-key')
            with self.assertRaisesRegex(BudgetError, 'отключены'):
                paid.complete(MESSAGES)
        self.assertFalse(self.path.exists())
        self.assertFalse(spending.status(self.path)['enabled'])
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 10, 'interactive_answer')

    def test_new_installation_binds_account_on_first_reservation(self):
        self.configure()
        self.assertIsNone(json.loads(self.path.read_text())['account_id'])
        self.ledger.reserve(MESSAGES, 10, 'interactive_answer')
        self.assertEqual(self.ledger.read()['account_id'], 'test')
        with self.assertRaises(BudgetError):
            BudgetLedger(self.path, 'different-account').read()

    def test_count_limit_survives_restart(self):
        self.configure()
        for _ in range(3):
            entry, _ = self.ledger.reserve(MESSAGES, 10, 'interactive_answer')
            self.ledger.complete(entry, {'prompt_tokens': 2, 'completion_tokens': 2})
        with self.assertRaises(BudgetError):
            BudgetLedger(self.path, 'test').reserve(MESSAGES, 10, 'interactive_answer')
        state = spending.status(self.path)
        self.assertEqual((state['used_calls'], state['remaining_calls']), (3, 0))

    def test_money_limit_checked_before_reservation(self):
        self.configure(amount='0.000001')
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 2048, 'interactive_answer')
        self.assertEqual(self.ledger.read()['entries'], [])

    def test_editing_limits_and_disabling_never_resets_usage(self):
        self.configure()
        entry, _ = self.ledger.reserve(MESSAGES, 10, 'interactive_answer')
        self.ledger.complete(entry, {'prompt_tokens': 2, 'completion_tokens': 2})
        before = copy.deepcopy(self.ledger.read()['entries'])
        self.configure(amount='0.20', calls='4', enabled=False)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 10, 'interactive_answer')
        self.configure(amount='0.30', calls='5')
        state = spending.status(self.path)
        self.assertEqual(state['used_calls'], 1)
        self.assertEqual(state['remaining_calls'], 4)
        self.assertEqual(before, self.ledger.read()['entries'])
        self.assertEqual(len(self.ledger.read()['ui_budget']['changes']), 3)

    def test_pending_cost_kept_and_increasing_limit_does_not_unblock(self):
        self.configure()
        _, reserve = self.ledger.reserve(MESSAGES, 10, 'interactive_answer')
        self.configure(amount='1', calls='30')
        state = spending.status(self.path)
        self.assertEqual(state['used_nano'], reserve)
        self.assertEqual(state['blocking_calls'], 1)
        with self.assertRaisesRegex(BudgetError, 'неопределённую'):
            self.ledger.reserve(MESSAGES, 10, 'interactive_answer')

    def test_prior_history_is_preserved_and_visible_at_first_setup(self):
        data = self.ledger.read()
        data['entries'] = [dict(id=0, status='pending', reserved_nano=12345)]
        self.path.write_text(json.dumps(data))
        before = copy.deepcopy(data['entries'])
        self.configure()
        state = spending.status(self.path)
        self.assertEqual(state['historical_nano'], 12345)
        self.assertEqual(state['historical_calls'], 1)
        self.assertEqual(state['uncertain_nano'], 12345)
        self.assertEqual(state['used_calls'], 0)
        self.assertEqual(self.ledger.read()['entries'], before)
        # The first setup is an explicit allowance for future work, not deletion of old reservations.
        self.assertEqual(self.ledger.reserve(MESSAGES, 10, 'interactive_answer')[0], 1)

    def test_history_tampering_is_rejected(self):
        data = self.ledger.read()
        data['entries'] = [dict(id=0, status='complete', cost_nano=100, reserved_nano=100)]
        self.path.write_text(json.dumps(data))
        self.configure()
        data = self.ledger.read()
        data['entries'][0]['cost_nano'] = 0
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.read()

    def test_localized_amounts_and_invalid_inputs(self):
        self.assertEqual(self.configure(amount='0,10')['limit_nano'], 100_000_000)
        for value in ('NaN', 'Infinity', '-1', 'word'):
            with self.assertRaises(spending.SpendingError):
                self.configure(amount=value)
        for value in ('3.5', '-1', '', True):
            with self.assertRaises(spending.SpendingError):
                self.configure(calls=value)
        with self.assertRaises(spending.SpendingError):
            self.configure(amount='0')

    def test_changed_floor_during_balance_check_is_honored(self):
        self.configure(floor='0')
        def handler(request):
            self.assertEqual(request.method, 'GET')
            self.configure(floor='5')
            return httpx.Response(200, json={'balance_infos': [{'currency': 'USD', 'total_balance': '4'}]})
        with httpx.Client(base_url='https://api.deepseek.com', transport=httpx.MockTransport(handler)) as client:
            paid = BudgetedDeepSeek(self.path, client=client, key='test-key')
            with self.assertRaisesRegex(BudgetError, 'остатка'):
                paid.complete(MESSAGES)
        self.assertEqual(len(spending.read_ledger(self.path)['entries']), 0)

    def test_cached_answer_available_while_paid_requests_disabled(self):
        cache_root = Path(self.temp.name) / 'cache'
        key = rag_pilot.digest([rag_pilot.MODEL, MESSAGES, 10, 'non-thinking-v1'])
        path = cache_root / 'responses' / (key + '.json')
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'text': 'Existing answer'}))
        def handler(request):
            self.fail('Cached answer must not call a provider')
        with httpx.Client(base_url='https://api.deepseek.com', transport=httpx.MockTransport(handler)) as client:
            paid = BudgetedDeepSeek(self.path, client=client, key='test-key')
            with patch.object(rag_pilot, 'PILOT_ROOT', cache_root):
                response = rag_pilot.cached_complete(paid, MESSAGES, 10, 'interactive_answer')
        self.assertTrue(response['cached'])
        self.assertFalse(self.path.exists())
