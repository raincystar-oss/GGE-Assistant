"""Import one user-selected recording using the existing pipeline, without paid LLM calls."""
from dataclasses import dataclass, field
from datetime import date
import json
import re
from urllib.parse import urlsplit

from .anythingllm import AnythingLLMClient
from .config import load_config
from .db import connect, init_db
from .lock import AppLock, LockError
from .processing import ProcessingContext, process_material
from .rag_context import ContextConfig, safe_card
from .rag_pilot import PILOT_ROOT, get_hits, prepare_index
from .state import add_material

PHASES = {
    'checking': 'Проверяю доступность базы…',
    'registering': 'Регистрирую материал…',
    'audio': 'Получаю запись и подготавливаю аудио…',
    'transcribing': 'Распознаю речь локальной моделью…',
    'quality': 'Проверяю качество расшифровки…',
    'chunking': 'Подготавливаю текстовые фрагменты…',
    'indexing': 'Добавляю документы в базу…',
    'verifying': 'Проверяю документы и поиск…',
}


class ImportProblem(RuntimeError):
    """Only safe, user-facing messages; never include raw provider errors."""


@dataclass(frozen=True)
class MaterialInput:
    media_url: str = field(repr=False)
    title: str
    webinar_date: str
    page_url: str = ''


def validate_input(media_url, title, webinar_date, page_url=''):
    media_url, title, webinar_date, page_url = (s.strip() for s in (media_url, title, webinar_date, page_url))
    try:
        parts = urlsplit(media_url)
        valid = (parts.scheme == 'https' and parts.netloc in ('rutube.ru', 'www.rutube.ru')
                 and re.fullmatch(r'/(?:video|play/embed)/[a-fA-F0-9]{32}/?', parts.path))
    except ValueError:
        valid = False
    if not valid:
        raise ImportProblem('Вставьте ссылку на запись RuTube вида https://rutube.ru/video/…/. '
                            'Ссылка на страницу мероприятия указывается в отдельном поле.')
    if not title or len(title) > 300 or '\n' in title:
        raise ImportProblem('Укажите название материала одной строкой, до 300 символов.')
    try:
        parsed = date.fromisoformat(webinar_date)
        if parsed.isoformat() != webinar_date:
            raise ValueError()
    except ValueError:
        raise ImportProblem('Укажите дату в формате ГГГГ-ММ-ДД, например 2026-06-30.') from None
    if page_url and (page_url != safe_card(page_url) or '?' in page_url or '#' in page_url):
        raise ImportProblem('Для страницы мероприятия используйте публичную ссылку edu.gge.ru или kb.gge.ru без параметров доступа.')
    return MaterialInput(media_url, title, webinar_date, page_url)


def _client(app):
    return AnythingLLMClient(base_url=app.anythingllm_base_url,
        api_key_path=app.anythingllm_api_key_path, workspace_slug=app.anythingllm_workspace_slug,
        max_attempts=1)


def import_material(item, config=None, progress=lambda stage: None):
    item = validate_input(item.media_url, item.title, item.webinar_date, item.page_url)
    app = load_config(config)
    if app.use_test_handlers:
        raise ImportProblem('В выбранной конфигурации включён тестовый режим. Добавление реальных записей отключено.')
    try:
        with AppLock(app.lock_path), AppLock(PILOT_ROOT / 'pilot.lock'):
            progress('checking')
            with _client(app) as client:
                client.workspace()  # Existing workspace only; never reset its settings.
            conn = connect(app.db_path)
            try:
                init_db(conn)
                progress('registering')
                material_id = add_material(conn, media_url=item.media_url, title=item.title,
                    webinar_date=item.webinar_date, discovery_url=item.page_url or None)
                status = process_material(conn, material_id, ProcessingContext(
                    data_root=app.data_root, config=app, use_test_handlers=False,
                    preserve_workspace_settings=True, progress=progress))
                if status != 'COMPLETE':
                    raise ImportProblem('Обработка не завершена. Готовые этапы сохранены. '
                        'Требуется проверка записи перед повторным запуском; автоматического повтора загрузки нет.')
                progress('verifying')
                rows = conn.execute('SELECT * FROM chunks WHERE material_id=?', (material_id,)).fetchall()
                with _client(app) as client:
                    paths = {d['docpath'] for d in client.workspace().get('documents', [])}
                    if not rows or any(r['status'] != 'COMPLETE' or r['document_location'] not in paths for r in rows):
                        raise ImportProblem('Не удалось подтвердить связь всех документов с выбранной базой. Требуется проверка.')
                    index = None
                    try:
                        index, stats = prepare_index(app, client, ContextConfig())
                        if material_id not in stats['materials']:
                            raise ImportProblem('Материал не появился в поисковом индексе. Требуется проверка.')
                        chunks = json.loads((app.data_root / 'materials' / material_id / 'chunks.json').read_text(encoding='utf-8'))
                        hits = get_hits(client, chunks['chunks'][0]['text'][:400], 8)
                        titles = {r['document_title'].removesuffix('.txt') for r in rows}
                        if not any(str(h.get('metadata', {}).get('title', '')).removesuffix('.txt') in titles for h in hits):
                            raise ImportProblem('Документы загружены, но контрольный поиск их не подтвердил. Требуется проверка.')
                    finally:
                        if index:
                            index.close()
                transcript = json.loads((app.data_root / 'materials' / material_id / 'transcript.json').read_text(encoding='utf-8'))
                return {'material_id': material_id, 'partial': transcript.get('quality', {}).get('status') == 'partial'}
            finally:
                conn.close()
    except ImportProblem:
        raise
    except LockError:
        raise ImportProblem('Уже выполняется другая операция с базой. Дождитесь её завершения.') from None
    except Exception:
        raise ImportProblem('Не удалось завершить обработку. Проверьте запуск AnythingLLM, доступность записи, '
                            'FFmpeg и настройку локальной модели. Готовые результаты сохранены.') from None
