import json
from pathlib import Path
import tempfile
import unittest

import httpx

from gge_assistant.rag_budget import BudgetedDeepSeek, BudgetError
from gge_assistant.request_trace import RequestTrace
from gge_assistant.spending import configure


class TraceTests(unittest.TestCase):
    def run_request(self, handler):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)
        client = httpx.Client(base_url='https://api.deepseek.com',
                             transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        paid = BudgetedDeepSeek(path / 'budget.json', client=client, key='SECRET_KEY')
        configure(path / 'budget.json', '0.50', '30', True, '3.50')
        return path, paid

    def rows(self, path):
        files = list((path / 'request_traces').glob('*.jsonl'))
        self.assertEqual(len(files), 1)
        return [json.loads(line) for line in files[0].read_text().splitlines()]

    def test_success_and_secret_exclusion(self):
        def handler(request):
            if request.url.path == '/user/balance':
                return httpx.Response(200, json={'balance_infos': [{'currency': 'USD', 'total_balance': '5'}]})
            return httpx.Response(200, headers={'x-request-id': 'SECRET_HEADER'}, json={
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                'choices': [{'message': {'content': 'SECRET_RESPONSE'}, 'finish_reason': 'stop'}]})
        path, paid = self.run_request(handler)
        paid.complete([{'role': 'system', 'content': 'SECRET_PROMPT'},
                       {'role': 'user', 'content': 'SECRET_QUESTION'}], max_output=10)
        rows = self.rows(path)
        self.assertEqual(rows[-1]['stage'], 'complete')
        self.assertEqual(next(r['http_status'] for r in rows if r['stage'] == 'response_headers'), 200)
        self.assertNotIn('SECRET', json.dumps(rows))

    def test_failure_retains_reservation_and_logs_status(self):
        def handler(request):
            if request.url.path == '/user/balance':
                return httpx.Response(200, json={'balance_infos': [{'currency': 'USD', 'total_balance': '5'}]})
            return httpx.Response(503, text='SECRET_ERROR')
        path, paid = self.run_request(handler)
        with self.assertRaises(BudgetError):
            paid.complete([{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'test'}])
        self.assertEqual(paid.ledger.read()['entries'][0]['status'], 'pending')
        rows = self.rows(path)
        self.assertEqual(rows[-1]['stage'], 'request_failed')
        self.assertEqual(next(r['http_status'] for r in rows if r['stage'] == 'response_headers'), 503)
        self.assertNotIn('SECRET', json.dumps(rows))

    def test_network_callback_discards_sensitive_info(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = RequestTrace(Path(directory))
            trace.network('http11.send_request_body.complete', {'headers': 'SECRET_KEY'})
            trace.network('unknown.SECRET', {'body': 'SECRET'})
            self.assertNotIn('SECRET', trace.path.read_text())
            self.assertEqual(len(trace.path.read_text().splitlines()), 1)

    def test_timeout_and_budget_block_are_distinguishable(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == '/user/balance':
                return httpx.Response(200, json={'balance_infos': [{'currency': 'USD', 'total_balance': '5'}]})
            raise httpx.ReadTimeout('SECRET_ERROR', request=request)
        path, paid = self.run_request(handler)
        messages = [{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'test'}]
        with self.assertRaises(BudgetError):
            paid.complete(messages)
        self.assertEqual(self.rows(path)[-1]['stage'], 'timeout')
        self.assertEqual(paid.ledger.read()['entries'][0]['status'], 'pending')
        with self.assertRaises(BudgetError):
            paid.complete(messages)
        self.assertEqual(calls.count('/chat/completions'), 1)
        logs = [p.read_text() for p in (path / 'request_traces').glob('*.jsonl')]
        self.assertTrue(any('stopped_before_send' in log for log in logs))
        self.assertNotIn('SECRET_ERROR', ''.join(logs))
