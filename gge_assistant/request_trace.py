"""Durable request stages only; never persist bodies, URLs, headers or errors."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RequestTrace:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (uuid4().hex + '.jsonl')
        self.started = time.monotonic()

    def event(self, stage: str, **numbers):
        # Metadata is deliberately limited to numeric measurements and identifiers.
        if any(type(value) not in (int, float) for value in numbers.values()):
            raise ValueError('Trace accepts numeric metadata only')
        row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=round(time.monotonic() - self.started, 3), **numbers)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def network(self, name, info):
        # Ignore info entirely: it can contain credentials, headers and exceptions.
        allowed = {
            'connection.connect_tcp.started', 'connection.connect_tcp.complete',
            'connection.start_tls.started', 'connection.start_tls.complete',
            'http11.send_request_headers.complete', 'http11.send_request_body.complete',
            'http11.receive_response_headers.started', 'http11.receive_response_headers.complete',
            'http2.send_request_headers.complete', 'http2.send_request_body.complete',
            'http2.receive_response_headers.started', 'http2.receive_response_headers.complete',
        }
        if name in allowed:
            self.event(name)
