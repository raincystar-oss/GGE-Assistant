from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from gge_assistant.source_links import SourceLinks, public_page


class SourceLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'materials.sqlite'
        with closing(sqlite3.connect(self.db)) as db:
            db.execute('CREATE TABLE materials (material_id TEXT, discovery_url TEXT)')
            db.execute('INSERT INTO materials VALUES (?,?)', ('a', 'https://kb.gge.ru/a'))
            db.commit()
        self.catalog = self.root / 'links.json'
        self.catalog.write_text(json.dumps({'b': {'url': 'https://edu.gge.ru/b'}}))

    def links(self):
        with patch('gge_assistant.source_links.load_config', return_value=SimpleNamespace(db_path=self.db)):
            return SourceLinks(catalog=self.catalog)

    def test_distinct_materials_same_title_stay_separate(self):
        links = self.links()
        sources = [dict(material_id=m, title='Same title', date='2026-01-01', source_id=s, start=0, end=1)
                   for m, s in [('a', 'S1'), ('b', 'S2'), ('a', 'S3')]]
        groups = links.groups(sources)
        self.assertEqual([g['url'] for g in groups], ['https://kb.gge.ru/a', 'https://edu.gge.ru/b'])
        self.assertEqual([f['source_id'] for f in groups[0]['fragments']], ['S1', 'S3'])

    def test_future_material_works_from_registry_without_code_changes(self):
        with closing(sqlite3.connect(self.db)) as db:
            db.execute('INSERT INTO materials VALUES (?,?)', ('new', 'https://kb.gge.ru/new'))
            db.commit()
        self.assertEqual(self.links().resolve({'material_id': 'new'}), 'https://kb.gge.ru/new')

    def test_missing_link_never_borrows_other_material_link(self):
        self.assertIsNone(self.links().resolve({'material_id': 'missing', 'title': 'Same title'}))

    def test_access_parameters_credentials_and_foreign_hosts_rejected(self):
        for value in ('https://rutube.ru/video/id/?p=secret', 'https://kb.gge.ru/a?p=secret',
                      'https://user:pass@kb.gge.ru/a', 'javascript:alert(1)', 'https://other.example/a'):
            self.assertFalse(public_page(value))
