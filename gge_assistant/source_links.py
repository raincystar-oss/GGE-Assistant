from contextlib import closing
"""Source-page metadata shared by chat and inspection tools; no model calls."""
import argparse
import json
import sqlite3
import re
from urllib.parse import urlsplit
from pathlib import Path

from .artifacts import atomic_write_json
from .config import load_config
from .rag_context import safe_card

CATALOG = Path(__file__).resolve().parent.parent / 'data' / 'source_pages.json'


def public_page(value):
    # Reject access parameters rather than silently turning a protected URL into a broken link.
    if not isinstance(value, str) or '?' in value or '#' in value:
        return ''
    card = safe_card(value)
    if card:
        return card
    try:
        parts = urlsplit(value)
        if (parts.scheme == 'https' and parts.netloc == 'rutube.ru'
                and re.fullmatch(r'/video/[a-fA-F0-9]{32}/?', parts.path)):
            return value
    except ValueError:
        pass
    return ''


class SourceLinks:
    def __init__(self, config=None, catalog=CATALOG):
        self.app = load_config(config)
        self.pages = json.loads(catalog.read_text(encoding='utf-8')) if catalog.exists() else {}
        self.materials = {}
        if self.app.db_path.exists():
            with closing(sqlite3.connect(self.app.db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                db.row_factory = sqlite3.Row
                self.materials = {r['material_id']: dict(r) for r in db.execute('SELECT * FROM materials')}

    def resolve(self, source):
        material_id = source.get('material_id')
        material = self.materials.get(material_id, {})
        saved = self.pages.get(material_id, {})
        # Current registry metadata takes precedence over old cached answer metadata.
        for candidate in (material.get('discovery_url'), saved.get('url'), source.get('url'), material.get('media_url')):
            url = public_page(candidate)
            if url:
                return url
        return None

    def groups(self, sources):
        groups = {}
        for source in sources:
            # Never join different materials just because titles or dates happen to match.
            key = source.get('material_id') or (source.get('title'), source.get('date'), source.get('url'))
            if key not in groups:
                groups[key] = dict(title=source.get('title', ''), date=source.get('date'),
                                   url=self.resolve(source), fragments=[])
            fragment = {k: source.get(k) for k in ('source_id', 'start', 'end')}
            if fragment not in groups[key]['fragments']:
                groups[key]['fragments'].append(fragment)
        return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description='Attach a verified public source page to a material')
    parser.add_argument('--config')
    parser.add_argument('--material-id', required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    links = SourceLinks(args.config)
    if args.material_id not in links.materials or not public_page(args.url):
        parser.error('Known material ID and public GGE page or RuTube video without access parameters required')
    material = links.materials[args.material_id]
    links.pages[args.material_id] = dict(url=args.url, title=material.get('title'), date=material.get('webinar_date'))
    atomic_write_json(CATALOG, links.pages)
    print('Source page saved; no reindexing or model call required.')


if __name__ == '__main__':
    main()
