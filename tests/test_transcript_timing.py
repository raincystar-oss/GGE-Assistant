from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch, MagicMock

from gge_assistant.config import load_config
from gge_assistant.transcription import _screen_timestamps, transcribe_material


class TimingTests(unittest.TestCase):
    def test_invalid_intervals_excluded_without_moving_valid_text(self):
        valid = {'start': 1, 'end': 10, 'text': 'Good phrase'}
        end = {'start': 20, 'end': 30, 'text': 'Another phrase'}
        source = {'duration': 30, 'segments': [valid,
            {'start': 12, 'end': 50, 'text': 'Outside recording'},
            {'start': 11, 'end': 9, 'text': 'Reversed'}, end]}
        result, issues = _screen_timestamps(source)
        self.assertEqual(result['segments'], [valid, end])
        self.assertEqual(len(issues), 2)
        self.assertEqual(len(source['segments']), 4)

    def test_overlapping_and_nonfinite_timestamps_quarantined(self):
        result, issues = _screen_timestamps({'duration': 30, 'segments': [
            {'start': 0, 'end': 10, 'text': 'Good'},
            {'start': 9, 'end': 15, 'text': 'Overlap'},
            {'start': float('nan'), 'end': 20, 'text': 'Bad'}]})
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(len(issues), 2)

    def test_raw_pass_retained_and_final_marked_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / 'input.wav'
            audio.write_bytes(b'synthetic-audio-placeholder')
            config = replace(load_config(), data_root=root, asr_quality_check=False,
                             min_transcript_chars=10, keep_audio=True)
            model = MagicMock()
            model.transcribe.return_value = (iter([
                SimpleNamespace(start=0, end=10, text='A sufficiently long valid phrase'),
                SimpleNamespace(start=50, end=80, text='Timestamp beyond recording')]), SimpleNamespace(language_probability=1))
            output = root / 'materials' / 'test' / 'transcript.json'
            material = {'material_id': 'test', 'platform_key': 'test', 'title': 'Synthetic test',
                        'webinar_date': None, 'published_at': None}
            with patch('gge_assistant.transcription._ensure_audio', return_value=audio), \
                 patch('gge_assistant.transcription._probe_duration', return_value=60), \
                 patch('gge_assistant.transcription._add_nvidia_dll_directories'), \
                 patch('faster_whisper.WhisperModel', return_value=model):
                transcribe_material(config, material, output)
            raw = json.loads((output.parent / 'transcript_primary_raw.json').read_text())
            final = json.loads(output.read_text())
            self.assertEqual(len(raw['segments']), 2)
            self.assertEqual(len(final['segments']), 1)
            self.assertEqual(final['quality']['status'], 'partial')

    def test_two_pass_quality_report_includes_timing_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / 'input.wav'
            with wave.open(str(audio), 'wb') as stream:
                stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                stream.writeframes(b'\0\0' * 16000 * 30)
            config = replace(load_config(), data_root=root, asr_quality_check=True,
                             min_transcript_chars=10, keep_audio=True)
            good = SimpleNamespace(start=0, end=10, text='A sufficiently long stable phrase')
            bad = SimpleNamespace(start=25, end=50, text='Timestamp beyond recording')
            model = MagicMock()
            info = SimpleNamespace(language_probability=1)
            model.transcribe.side_effect = [(iter([good, bad]), info), (iter([good]), info)]
            output = root / 'materials' / 'test' / 'transcript.json'
            material = {'material_id': 'test', 'platform_key': 'test', 'title': 'Synthetic test',
                        'webinar_date': None, 'published_at': None}
            phases = []
            with patch('gge_assistant.transcription._ensure_audio', return_value=audio), \
                 patch('gge_assistant.transcription._probe_duration', return_value=30), \
                 patch('gge_assistant.transcription._add_nvidia_dll_directories'), \
                 patch('faster_whisper.WhisperModel', return_value=model):
                transcribe_material(config, material, output, progress=phases.append)
            final = json.loads(output.read_text())
            report = json.loads((output.parent / 'asr_quality.json').read_text())
            self.assertIn('quality', phases)
            self.assertEqual(model.transcribe.call_count, 2)
            self.assertEqual(final['quality']['status'], 'partial')
            self.assertEqual(len(report['timestamp_checks']['primary']), 1)
            self.assertTrue((output.parent / 'transcript_second_pass_raw.json').exists())
