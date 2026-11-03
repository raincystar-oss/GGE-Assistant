from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .db import connect, init_db
from .lock import AppLock, LockError
from .logging_setup import configure_logging
from .processing import ProcessingContext, process_all, process_material, retry_stage
from .state import add_material, get_material, overall_status

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gge-assistant")
    parser.add_argument("--config", help="Path to config.ini")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-url", help="Register a material URL")
    add.add_argument("url")
    add.add_argument("--platform-key")
    add.add_argument("--discovery-url")
    add.add_argument("--title")
    add.add_argument("--webinar-date")
    add.add_argument("--published-at")

    run = sub.add_parser("run", help="Process all queued materials")
    _add_test_flags(run)

    process = sub.add_parser("process", help="Process one material")
    process.add_argument("--material-id", required=True)
    _add_test_flags(process)

    retry = sub.add_parser("retry", help="Retry one unfinished or failed stage")
    retry.add_argument("--material-id", required=True)
    retry.add_argument("--stage", required=True, choices=["transcript", "chunks"])
    _add_test_flags(retry)

    status = sub.add_parser("status", help="Show one material status")
    status.add_argument("--material-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.data_root.mkdir(parents=True, exist_ok=True)
    configure_logging(config.log_path, args.verbose)

    try:
        with AppLock(config.lock_path):
            conn = connect(config.db_path)
            try:
                init_db(conn)
                return _run_command(conn, config, args)
            finally:
                conn.close()
    except LockError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_command(conn, config, args) -> int:
    context = ProcessingContext(
        data_root=config.data_root,
        config=config,
        use_test_handlers=config.use_test_handlers or getattr(args, "test_handlers", False),
        crash_after_file=getattr(args, "test_crash_after_file", False),
        crash_after_json_before_txt=getattr(args, "test_crash_after_json_before_txt", False),
        fail_stage_for_title=getattr(args, "test_fail_stage_for_title", None),
        test_lose_upload_response_once=getattr(args, "test_lose_upload_response_once", False),
    )
    if args.command == "add-url":
        material_id = add_material(
            conn,
            media_url=args.url,
            platform_key=args.platform_key,
            discovery_url=args.discovery_url,
            title=args.title,
            webinar_date=args.webinar_date,
            published_at=args.published_at,
        )
        print(material_id)
        return 0
    if args.command == "run":
        results = process_all(conn, context)
        for material_id, status in results.items():
            print(f"{material_id} {status}")
        return 0 if all(status != "ERROR" for status in results.values()) else 1
    if args.command == "process":
        status = process_material(conn, args.material_id, context)
        print(f"{args.material_id} {status}")
        return 0 if status != "ERROR" else 1
    if args.command == "retry":
        status = retry_stage(conn, args.material_id, args.stage, context)
        print(f"{args.material_id} {status}")
        return 0 if status != "ERROR" else 1
    if args.command == "status":
        material = get_material(conn, args.material_id)
        if material is None:
            print(f"Unknown material_id: {args.material_id}", file=sys.stderr)
            return 1
        print(f"{args.material_id} {overall_status(conn, args.material_id)}")
        return 0
    raise AssertionError(args.command)


def _add_test_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-handlers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-crash-after-file", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-crash-after-json-before-txt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-fail-stage-for-title", help=argparse.SUPPRESS)
    parser.add_argument("--test-lose-upload-response-once", action="store_true", help=argparse.SUPPRESS)
