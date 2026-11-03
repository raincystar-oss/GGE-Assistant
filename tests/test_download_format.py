import unittest
from yt_dlp import YoutubeDL
from gge_assistant.transcription import DOWNLOAD_FORMAT


class DownloadFormatTests(unittest.TestCase):
    def select(self, heights, audio=False):
        formats = [dict(format_id=str(h), height=h, width=h * 2, tbr=h,
                        url=f'https://example.com/{h}.mp4', ext='mp4', vcodec='avc1', acodec='mp4a') for h in heights]
        if audio:
            formats.append(dict(format_id='audio', url='https://example.com/audio.m4a',
                                ext='m4a', vcodec='none', acodec='mp4a', abr=128))
        with YoutubeDL({'format': DOWNLOAD_FORMAT, 'quiet': True, 'no_warnings': True}) as downloader:
            return downloader.process_ie_result(dict(id='test', title='Test format selection', formats=formats), download=False)['format_id']

    def test_audio_only_preferred(self):
        self.assertEqual(self.select([232, 360, 1080], audio=True), 'audio')

    def test_without_audio_track_avoids_hd(self):
        self.assertEqual(self.select([232, 360, 1080]), '360')

    def test_high_resolution_only_source_still_supported(self):
        self.assertEqual(self.select([1080]), '1080')
