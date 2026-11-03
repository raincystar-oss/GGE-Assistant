from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gge_assistant import material_import as importer
from gge_assistant.config import load_config
from gge_assistant.db import connect
from gge_assistant.processing import process_material
from gge_assistant.processing import _ingest_chunks, ProcessingContext


URL = 'https://rutube.ru/video/' + 'a' * 32 + '/'


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = replace(load_config(), data_root=self.root, db_path=self.root / 'registry.sqlite',
                           lock_path=self.root / 'app.lock', use_test_handlers=False)
        self.item = importer.validate_input(URL, 'Test recording', '2026-11-03', 'https://kb.gge.ru/test')
        self.client = MagicMock()
        self.client.workspace_slug = self.app.anythingllm_workspace_slug
        self.client.__enter__.return_value = self.client
        self.client.workspace.return_value = {'documents': [{'docpath': 'custom-documents/test.json'}]}
        self.patches = [patch.object(importer, 'load_config', return_value=self.app),
                        patch.object(importer, '_client', return_value=self.client),
                        patch.object(importer, 'PILOT_ROOT', self.root / 'pilot'),
                        patch('gge_assistant.rag_pilot.PILOT_ROOT', self.root / 'pilot'),
                        patch.object(importer, 'get_hits', return_value=[{'metadata': {'title': 'test.txt'}}]),
                        patch.object(importer, 'process_material', side_effect=self.fake_pipeline)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_pipeline(self, conn, material_id, context):
        self.assertTrue(context.preserve_workspace_settings)
        self.assertFalse(context.use_test_handlers)
        # Isolated synthetic handlers stand in for ASR/upload, not a real webinar.
        status = process_material(conn, material_id, replace(context, use_test_handlers=True))
        conn.execute("UPDATE chunks SET status='COMPLETE', document_title='test', "
                     "document_location='custom-documents/test.json' WHERE material_id=?", (material_id,))
        conn.commit()
        return status

    def test_import_verifies_links_and_search_and_preserves_duplicate(self):
        phases = []
        first = importer.import_material(self.item, progress=phases.append)
        second = importer.import_material(self.item)
        self.assertEqual(first['material_id'], second['material_id'])
        self.assertIn('verifying', phases)
        self.assertFalse(first['partial'])
        self.client.ensure_workspace.assert_not_called()
        with self.app.db_path.open('rb'):
            pass
        conn = connect(self.app.db_path)
        try:
            self.assertEqual(conn.execute('SELECT count(*) FROM materials').fetchone()[0], 1)
        finally:
            conn.close()

    def test_success_not_reported_if_workspace_link_missing(self):
        self.client.workspace.return_value = {'documents': []}
        with self.assertRaisesRegex(importer.ImportProblem, 'связь'):
            importer.import_material(self.item)

    def test_success_not_reported_if_vector_search_misses(self):
        with patch.object(importer, 'get_hits', return_value=[]):
            with self.assertRaisesRegex(importer.ImportProblem, 'поиск'):
                importer.import_material(self.item)

    def test_unavailable_workspace_stops_before_registering_and_hides_secret(self):
        self.client.workspace.side_effect = RuntimeError('secret_token')
        with self.assertRaises(importer.ImportProblem) as error:
            importer.import_material(self.item)
        self.assertNotIn('secret_token', str(error.exception))
        self.assertFalse(self.app.db_path.exists())

    def test_form_validation_and_private_url_not_in_repr(self):
        for url in ('file:///tmp/example', 'https://kb.gge.ru/card', 'https://evil.example/video/' + 'a' * 32):
            with self.assertRaises(importer.ImportProblem):
                importer.validate_input(url, 'Title', '2026-11-03')
        for date in ('2026-02-30', '03.11.2026', ''):
            with self.assertRaises(importer.ImportProblem):
                importer.validate_input(URL, 'Title', date)
        private = importer.validate_input(URL + '?p=secret_token', 'Title', '2026-11-03')
        self.assertNotIn('secret_token', repr(private))

    def test_error_or_ambiguous_pipeline_is_not_retried(self):
        with patch.object(importer, 'process_material', return_value='NEEDS_ATTENTION') as process:
            with self.assertRaisesRegex(importer.ImportProblem, 'автоматического повтора'):
                importer.import_material(self.item)
            self.assertEqual(process.call_count, 1)

    def test_ingestion_reuses_existing_workspace_without_updating_settings(self):
        result = importer.import_material(self.item)
        material_id = result['material_id']
        directory = self.root / 'materials' / material_id
        conn = connect(self.app.db_path)
        try:
            material = conn.execute('SELECT * FROM materials WHERE material_id=?', (material_id,)).fetchone()
            phases = []
            with patch('gge_assistant.processing.AnythingLLMClient', return_value=self.client):
                _ingest_chunks(conn, material, ProcessingContext(
                    data_root=self.root, config=self.app, use_test_handlers=False,
                    preserve_workspace_settings=True, progress=phases.append),
                    directory / 'chunks.json', directory / 'transcript.json')
            self.client.ensure_workspace.assert_not_called()
            self.client.upload_or_locate_document.assert_not_called()
            self.assertIn('indexing', phases)
        finally:
            conn.close()
