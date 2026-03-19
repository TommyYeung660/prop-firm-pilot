"""
Pack production logs, data artifacts, and LLM summaries for PropFirmPilot.

Collects raw production outputs, exports Telegram messages, generates
LLM-optimized summaries via gpt-5.4, writes bundle metadata, zips
everything into a single archive, and uploads the archive to Dropbox.

Usage:
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml \
        --version v1.4.2 --days 5
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml --no-summarize
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.ops.dropbox_artifacts import DropboxArtifactsClient
from src.version import get_app_version, get_release_tag

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://right.codes/codex/v1"
MODEL_NAME = "gpt-5.4"
MAX_LOG_CHARS = 100_000
MAX_LLM_INPUT_CHARS = 100_000


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack production logs and summaries")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML (merged with config/default.yaml)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=f"Version string like {get_release_tag()} (defaults to current release tag)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of data to include (default: 7)",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip LLM summarization and only collect raw files",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for the zip archive (default: .)",
    )
    return parser.parse_args()


def _normalize_release_tag(version: str) -> str:
    """Normalize a bare semantic version or v-prefixed tag to release-tag form."""
    normalized = version.strip()
    if not normalized:
        raise ValueError("Version must not be empty")
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    return normalized


def _resolve_version(version: str | None) -> str:
    """Resolve the effective packer version and fail fast on drift."""
    current_tag = get_release_tag()
    if version is None:
        return current_tag
    requested_tag = _normalize_release_tag(version)
    if requested_tag != current_tag:
        raise ValueError(
            f"Explicit version '{requested_tag}' does not match current project version "
            f"'{current_tag}' ({get_app_version()})"
        )
    return current_tag


# ── Config Loading ──────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: {}", path)
        return {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML {}: {}", path, exc)
        return {}
    return data


def _load_merged_config(config_path: Path) -> dict[str, Any]:
    default_path = Path("config/default.yaml")
    default_cfg = _load_yaml(default_path)
    account_cfg = _load_yaml(config_path)
    return _deep_merge(default_cfg, account_cfg)


def _get_config_value(config: dict[str, Any], keys: list[str], default: str) -> str:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return str(cursor)


def _resolve_account_name(config: dict[str, Any], config_path: Path) -> str:
    """Resolve account_name from config, falling back to the config file stem."""
    return _get_config_value(config, ["account_name"], config_path.stem)


def _build_dropbox_bundle_dir(account_name: str) -> str:
    """Build the Dropbox folder path for a given account's prod bundles."""
    return f"/prop-firm-pilot/prod_logs/{account_name}"


def _build_dropbox_bundle_path(account_name: str, zip_name: str) -> str:
    """Build the Dropbox file path for a bundle zip."""
    return f"{_build_dropbox_bundle_dir(account_name)}/{zip_name}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML file with stable formatting for bundle snapshots."""
    try:
        _ensure_dir(path.parent)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Wrote YAML snapshot {}", path)
    except OSError as exc:
        logger.error("Failed to write YAML {}: {}", path, exc)


def _write_config_snapshots(
    raw_config_dir: Path,
    default_config_path: Path,
    account_config_path: Path,
    merged_config: dict[str, Any],
) -> None:
    """Persist default, account, and merged YAML snapshots into the bundle."""
    _ensure_dir(raw_config_dir)
    _copy_file(default_config_path, raw_config_dir / default_config_path.name)
    _copy_file(account_config_path, raw_config_dir / account_config_path.name)
    _write_yaml(raw_config_dir / "merged_config.yaml", merged_config)


def _git_output(args: list[str]) -> str | None:
    """Run a small git command and return stripped stdout when available."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output or None


def _write_bundle_manifest(
    manifest_path: Path,
    account_name: str,
    config_path: str,
    version: str,
    app_version: str,
    generated_at_utc: str,
    days: int,
    date_range: str,
    bundle_folder: str,
    zip_name: str,
    git_commit: str | None,
    git_branch: str | None,
    included_logs: list[str],
    included_data_files: list[str],
    included_config_files: list[str],
    included_summary_files: list[str],
) -> None:
    """Write machine-readable bundle metadata for downstream tooling and LLMs."""
    payload = {
        "account_name": account_name,
        "config_path": config_path,
        "version": version,
        "app_version": app_version,
        "generated_at_utc": generated_at_utc,
        "days": days,
        "date_range": date_range,
        "bundle_folder": bundle_folder,
        "zip_name": zip_name,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "included_logs": included_logs,
        "included_data_files": included_data_files,
        "included_config_files": included_config_files,
        "included_summary_files": included_summary_files,
    }
    try:
        _ensure_dir(manifest_path.parent)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote bundle manifest {}", manifest_path)
    except OSError as exc:
        logger.error("Failed to write bundle manifest {}: {}", manifest_path, exc)


def _upload_bundle_zip(zip_path: Path, account_name: str) -> str:
    """Upload the generated bundle zip to Dropbox and return the remote path."""
    remote_path = _build_dropbox_bundle_path(account_name, zip_path.name)
    client = DropboxArtifactsClient()
    client.upload_file(zip_path, remote_path)
    return remote_path


# ── File Utilities ──────────────────────────────────────────────────────────


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _within_days(path: Path, cutoff: datetime) -> bool:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except FileNotFoundError:
        return False
    return mtime >= cutoff


def _copy_file(src: Path, dst: Path) -> None:
    try:
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        logger.info("Copied file {}", src)
    except FileNotFoundError:
        logger.warning("Missing file: {}", src)
    except OSError as exc:
        logger.error("Failed to copy {}: {}", src, exc)


def _copy_tree_filtered(src_dir: Path, dst_dir: Path, cutoff: datetime) -> None:
    if not src_dir.exists():
        logger.warning("Missing directory: {}", src_dir)
        return
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if not _within_days(path, cutoff):
            continue
        relative = path.relative_to(src_dir)
        _copy_file(path, dst_dir / relative)


def _read_text_file(path: Path, max_chars: int | None = None) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        logger.warning("Missing file: {}", path)
        return ""
    except OSError as exc:
        logger.error("Failed to read {}: {}", path, exc)
        return ""
    if max_chars is not None and len(content) > max_chars:
        return content[-max_chars:]
    return content


def _select_log_files(log_file: Path, cutoff: datetime) -> list[Path]:
    """Return deduplicated log files relevant to the bundle, ordered by mtime."""
    selected: list[Path] = []
    seen: set[Path] = set()
    if log_file.exists():
        selected.append(log_file)
        seen.add(log_file)
    if log_file.parent.exists():
        patterns = [f"{log_file.stem}_*.log*", f"{log_file.name}*"]
        for pattern in patterns:
            for path in log_file.parent.glob(pattern):
                if path.is_dir() or path in seen:
                    continue
                if not _within_days(path, cutoff):
                    continue
                selected.append(path)
                seen.add(path)
    return sorted(selected, key=lambda path: path.stat().st_mtime)


def _collect_logs(log_file: Path, logs_dir: Path, cutoff: datetime) -> None:
    selected_logs = _select_log_files(log_file, cutoff)
    if not selected_logs and not log_file.exists():
        logger.warning("Missing log file: {}", log_file)
    for path in selected_logs:
        _copy_file(path, logs_dir / path.name)
        if path == log_file and not _within_days(path, cutoff):
            logger.warning("Log file older than cutoff, still included: {}", log_file)


def _load_log_content(log_file: Path, cutoff: datetime, max_chars: int | None = None) -> str:
    """Load concatenated content from selected log files for summarization."""
    contents: list[str] = []
    for path in _select_log_files(log_file, cutoff):
        content = _read_text_file(path)
        if content:
            contents.append(content)
    combined = "\n\n".join(contents)
    if max_chars is not None and len(combined) > max_chars:
        return combined[-max_chars:]
    return combined


def _collect_data_files(raw_data_dir: Path, paths: Iterable[Path], cutoff: datetime) -> None:
    for path in paths:
        if not path.exists():
            logger.warning("Missing data file: {}", path)
            continue
        if _within_days(path, cutoff):
            _copy_file(path, raw_data_dir / path.name)
        else:
            _copy_file(path, raw_data_dir / path.name)
            logger.warning("Data file older than cutoff, still included: {}", path)


def _collect_memory(memory_dir: Path, raw_memory_dir: Path, cutoff: datetime) -> list[Path]:
    if not memory_dir.exists():
        logger.warning("Missing memory directory: {}", memory_dir)
        return []
    _ensure_dir(raw_memory_dir)
    selected: list[Path] = []
    for path in memory_dir.glob("*.md"):
        if not _within_days(path, cutoff):
            continue
        _copy_file(path, raw_memory_dir / path.name)
        selected.append(path)
    return selected


def _collect_eval_results(eval_dir: Path, raw_eval_dir: Path, cutoff: datetime) -> None:
    _copy_tree_filtered(eval_dir, raw_eval_dir, cutoff)


# ── Telegram Export ─────────────────────────────────────────────────────────


async def _export_telegram(raw_dir: Path) -> list[str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set; skipping Telegram export")
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.error("Telegram export failed: {}", exc)
        return []
    _ensure_dir(raw_dir)
    raw_path = raw_dir / "telegram_messages.json"
    try:
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved Telegram messages to {}", raw_path)
    except OSError as exc:
        logger.error("Failed to write Telegram messages: {}", exc)
    return _extract_telegram_texts(payload)


def _extract_telegram_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if not isinstance(payload, dict):
        return texts
    results = payload.get("result", [])
    if not isinstance(results, list):
        return texts
    for item in results:
        if not isinstance(item, dict):
            continue
        message = item.get("message") or item.get("edited_message")
        if not isinstance(message, dict):
            continue
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


# ── Summarization ───────────────────────────────────────────────────────────


async def _call_llm(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
) -> str:
    """Call gpt-5.4 via rightcodes OpenAI-compatible API."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = await client.post(url, json=payload, headers=headers, timeout=120.0)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _prepare_llm_input(content: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """Normalize and bound summary input before sending it to the LLM."""
    normalized = content.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def _has_meaningful_content(content: str) -> bool:
    """Return True when the content has any non-whitespace payload."""
    return bool(content and content.strip())


def _write_summary(path: Path, content: str) -> None:
    try:
        _ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote summary {}", path)
    except OSError as exc:
        logger.error("Failed to write summary {}: {}", path, exc)


async def _summarize_all(
    summary_dir: Path,
    log_content: str,
    trade_content: str,
    decisions_dump: str,
    memory_content: str,
    telegram_content: str,
) -> None:
    api_key = os.getenv("RIGHTCODE_API_KEY")
    prepared_log_content = _prepare_llm_input(log_content)
    prepared_trade_content = _prepare_llm_input(trade_content)
    prepared_decisions_dump = _prepare_llm_input(decisions_dump)
    prepared_memory_content = _prepare_llm_input(memory_content)
    prepared_telegram_content = _prepare_llm_input(telegram_content)
    if not api_key:
        logger.warning("RIGHTCODE_API_KEY not set; skipping LLM summaries")
        _write_summary(
            summary_dir / "log_summary.md",
            _build_placeholder_summary(
                "Log Summary (Fallback)",
                "LLM summary unavailable",
                prepared_log_content,
            ),
        )
        _write_summary(
            summary_dir / "trades_summary.md",
            _build_placeholder_summary(
                "Trades Summary (Fallback)",
                "LLM summary unavailable",
                prepared_trade_content,
            ),
        )
        _write_summary(
            summary_dir / "decisions_summary.md",
            _build_decisions_fallback_summary(prepared_trade_content),
        )
        _write_summary(
            summary_dir / "memory_summary.md",
            _build_placeholder_summary(
                "Memory Summary (Fallback)",
                "LLM summary unavailable",
                prepared_memory_content,
            ),
        )
        _write_summary(
            summary_dir / "telegram_summary.md",
            _build_telegram_fallback_summary(prepared_telegram_content),
        )
        return

    async with httpx.AsyncClient() as client:
        try:
            log_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個生產環境日誌分析專家。"
                "分析以下 prop-firm-pilot 交易系統日誌，生成結構化摘要。",
                prepared_log_content,
            )
            _write_summary(summary_dir / "log_summary.md", log_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize logs: {}", exc)
            _write_summary(
                summary_dir / "log_summary.md",
                _build_placeholder_summary(
                    "Log Summary (Fallback)",
                    f"LLM summary failed: {exc}",
                    prepared_log_content,
                ),
            )

        try:
            trades_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個交易績效分析專家。分析以下 JSONL 格式的交易日誌。",
                prepared_trade_content,
            )
            _write_summary(summary_dir / "trades_summary.md", trades_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize trades: {}", exc)
            _write_summary(
                summary_dir / "trades_summary.md",
                _build_placeholder_summary(
                    "Trades Summary (Fallback)",
                    f"LLM summary failed: {exc}",
                    prepared_trade_content,
                ),
            )

        if not _has_meaningful_content(prepared_decisions_dump):
            _write_summary(
                summary_dir / "decisions_summary.md",
                _build_decisions_fallback_summary(prepared_trade_content),
            )
        else:
            try:
                decisions_summary = await _call_llm(
                    client,
                    BASE_URL,
                    api_key,
                    "你是一個 AI 決策系統分析專家。分析以下 SQLite 決策數據庫的 dump。",
                    prepared_decisions_dump,
                )
                _write_summary(summary_dir / "decisions_summary.md", decisions_summary)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to summarize decisions: {}", exc)
                _write_summary(
                    summary_dir / "decisions_summary.md",
                    _build_decisions_fallback_summary(prepared_trade_content),
                )

        if not _has_meaningful_content(prepared_memory_content):
            _write_summary(
                summary_dir / "memory_summary.md",
                _build_placeholder_summary(
                    "Memory Summary (Fallback)",
                    "No memory content available",
                    prepared_memory_content,
                ),
            )
        else:
            try:
                memory_summary = await _call_llm(
                    client,
                    BASE_URL,
                    api_key,
                    "你是一個交易記憶分析專家。分析以下每日 Markdown 交易記憶日誌。",
                    prepared_memory_content,
                )
                _write_summary(summary_dir / "memory_summary.md", memory_summary)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to summarize memory: {}", exc)
                _write_summary(
                    summary_dir / "memory_summary.md",
                    _build_placeholder_summary(
                        "Memory Summary (Fallback)",
                        f"LLM summary failed: {exc}",
                        prepared_memory_content,
                    ),
                )

        if not _has_meaningful_content(prepared_telegram_content):
            _write_summary(
                summary_dir / "telegram_summary.md",
                _build_telegram_fallback_summary(prepared_telegram_content),
            )
        else:
            try:
                telegram_summary = await _call_llm(
                    client,
                    BASE_URL,
                    api_key,
                    "你是一個交易通知分析專家。分析以下 Telegram 交易通知歷史。",
                    prepared_telegram_content,
                )
                _write_summary(summary_dir / "telegram_summary.md", telegram_summary)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to summarize Telegram: {}", exc)
                _write_summary(
                    summary_dir / "telegram_summary.md",
                    _build_telegram_fallback_summary(prepared_telegram_content),
                )


# ── Data Extraction ─────────────────────────────────────────────────────────


def _load_trade_journal(path: Path, cutoff: datetime) -> str:
    if not path.exists():
        logger.warning("Missing trade journal: {}", path)
        return ""
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if _json_line_within_days(line, cutoff):
                    lines.append(line)
    except OSError as exc:
        logger.error("Failed to read trade journal {}: {}", path, exc)
        return ""
    return "\n".join(lines)


def _json_line_within_days(line: str, cutoff: datetime) -> bool:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, dict):
        return True
    candidates = [
        payload.get("timestamp"),
        payload.get("created_at"),
        payload.get("trade_time"),
        payload.get("open_time"),
        payload.get("closed_at"),
        payload.get("time"),
    ]
    for value in candidates:
        parsed = _parse_datetime(value)
        if parsed is None:
            continue
        return parsed >= cutoff
    return True


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _dump_sqlite_db(path: Path) -> str:
    if not path.exists():
        logger.warning("Missing decisions DB: {}", path)
        return ""
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as exc:
        logger.error("Failed to open SQLite DB {}: {}", path, exc)
        return ""
    conn.row_factory = sqlite3.Row
    dump_lines: list[str] = []
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for row in tables:
            table = row["name"]
            dump_lines.append(f"-- TABLE: {table}")
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            for record in rows:
                dump_lines.append(json.dumps(dict(record), ensure_ascii=False))
            dump_lines.append("")
    except sqlite3.Error as exc:
        logger.error("Failed to dump SQLite DB {}: {}", path, exc)
    finally:
        conn.close()
    return "\n".join(dump_lines)


def _load_memory_files(paths: list[Path]) -> str:
    contents: list[str] = []
    for path in paths:
        contents.append(_read_text_file(path))
    return "\n\n".join(contents)


# ── INDEX Generation ────────────────────────────────────────────────────────


def _collect_raw_file_listing(raw_dir: Path) -> list[str]:
    entries: list[str] = []
    if not raw_dir.exists():
        return entries
    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir():
            continue
        size = path.stat().st_size
        rel = path.relative_to(raw_dir)
        entries.append(f"- raw/{rel.as_posix()} ({size} bytes)")
    return entries


def _collect_summary_file_listing(summary_dir: Path) -> list[str]:
    entries: list[str] = []
    if not summary_dir.exists():
        return entries
    for path in sorted(summary_dir.glob("*.md")):
        if path.is_dir():
            continue
        size = path.stat().st_size
        rel = path.relative_to(summary_dir.parent)
        entries.append(f"- {rel.as_posix()} ({size} bytes)")
    return entries


def _collect_relative_paths(base_dir: Path, root_dir: Path) -> list[str]:
    """Collect sorted file paths relative to the bundle root."""
    if not base_dir.exists():
        return []
    return [
        path.relative_to(root_dir).as_posix()
        for path in sorted(base_dir.rglob("*"))
        if path.is_file()
    ]


def _parse_jsonl_objects(content: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _event_timestamp(payload: dict[str, Any]) -> datetime | None:
    candidates = [
        payload.get("timestamp"),
        payload.get("created_at"),
        payload.get("trade_time"),
        payload.get("open_time"),
        payload.get("closed_at"),
        payload.get("time"),
    ]
    for value in candidates:
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _event_day(payload: dict[str, Any]) -> str:
    parsed = _event_timestamp(payload)
    if parsed is not None:
        return parsed.astimezone(timezone.utc).date().isoformat()
    trade_date = payload.get("trade_date")
    if isinstance(trade_date, str):
        return trade_date
    return ""


def _build_placeholder_summary(title: str, reason: str, content: str = "") -> str:
    non_empty_lines = sum(1 for line in content.splitlines() if line.strip())
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- {reason}",
            f"- Non-empty lines: {non_empty_lines}",
        ]
    )


def _build_telegram_fallback_summary(telegram_content: str) -> str:
    messages = [line.strip() for line in telegram_content.splitlines() if line.strip()]
    lines = [
        "# Telegram Summary (Deterministic Fallback)",
        "",
        f"- Exported text messages: {len(messages)}",
    ]
    if not messages:
        lines.append("- No Telegram text messages were exported in this bundle.")
        return "\n".join(lines)
    lines.extend(["", "## Sample Messages"])
    lines.extend(f"- {message[:160]}" for message in messages[:5])
    return "\n".join(lines)


def _build_decisions_fallback_summary(trade_content: str) -> str:
    records = _parse_jsonl_objects(trade_content)
    event_counts = Counter(
        payload["type"]
        for payload in records
        if isinstance(payload.get("type"), str)
    )
    cancel_reasons = Counter(
        str(payload["reason"])
        for payload in records
        if payload.get("type") == "INTENT_CANCELLED" and payload.get("reason")
    )
    skip_reasons = Counter(
        str(payload["reason"])
        for payload in records
        if payload.get("type") == "SCANNER_SKIP" and payload.get("reason")
    )

    ordered_records = sorted(
        enumerate(records),
        key=lambda item: (
            _event_timestamp(item[1]) or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        ),
    )
    shadow_rows: dict[tuple[str, str], dict[str, int | float]] = {}

    def _format_counter_summary(counter: Counter) -> str:
        if not counter:
            return "none"
        return ",".join(f"{key}={count}" for key, count in sorted(counter.items()))

    for idx, (_, payload) in enumerate(ordered_records):
        event_type = payload.get("type")
        reason = payload.get("reason")
        symbol = payload.get("symbol")
        day = _event_day(payload)
        if event_type not in ("INTENT_CANCELLED", "SCANNER_SKIP"):
            continue
        if not isinstance(reason, str) or not isinstance(symbol, str) or not day:
            continue
        follow_up_events = [
            later_payload
            for _, later_payload in ordered_records[idx + 1 :]
            if later_payload.get("symbol") == symbol and _event_day(later_payload) == day
        ]
        opened = any(item.get("type") == "TRADE_OPENED" for item in follow_up_events)
        closed = [item for item in follow_up_events if item.get("type") == "TRADE_CLOSED"]
        pnl = sum(float(item.get("pnl", 0.0) or 0.0) for item in closed)
        trigger_sources = Counter(
            str(item.get("trigger_source", ""))
            for item in closed
            if item.get("trigger_source")
        )
        final_close_reasons = Counter(
            str(item.get("final_close_reason", item.get("reason", "")))
            for item in closed
            if item.get("final_close_reason") or item.get("reason")
        )
        key = (symbol, reason)
        bucket = shadow_rows.setdefault(
            key,
            {
                "count": 0,
                "same_day_follow_up_opens": 0,
                "same_day_follow_up_closes": 0,
                "same_day_follow_up_pnl": 0.0,
                "same_day_trigger_sources": Counter(),
                "same_day_final_close_reasons": Counter(),
            },
        )
        bucket["count"] += 1
        bucket["same_day_follow_up_opens"] += int(opened)
        bucket["same_day_follow_up_closes"] += int(bool(closed))
        bucket["same_day_follow_up_pnl"] += pnl
        bucket["same_day_trigger_sources"].update(trigger_sources)
        bucket["same_day_final_close_reasons"].update(final_close_reasons)

    lines = [
        "# Decisions Summary (Deterministic Fallback)",
        "",
        "## Lifecycle Counts",
        "| Event | Count |",
        "|------|------:|",
    ]
    for event_type in (
        "INTENT_CREATED",
        "INTENT_CANCELLED",
        "SCANNER_SKIP",
        "TRADE_OPENED",
        "TRADE_CLOSED",
    ):
        lines.append(f"| {event_type} | {event_counts.get(event_type, 0)} |")

    lines.extend(["", "## Cancellation Reasons", "| Reason | Count |", "|------|------:|"])
    if cancel_reasons:
        for reason, count in cancel_reasons.most_common():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Scanner Skip Reasons", "| Reason | Count |", "|------|------:|"])
    if skip_reasons:
        for reason, count in skip_reasons.most_common():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Shadow Analysis",
            "| Symbol | Reason | Events | Same-day Opens | Same-day Closes | Same-day PnL | Trigger Sources | Final Close Reasons |",
            "|------|------|------:|------:|------:|------:|------|------|",
        ]
    )
    if shadow_rows:
        for (symbol, reason), summary in sorted(shadow_rows.items()):
            lines.append(
                "| {} | {} | {} | {} | {} | {:.2f} | {} | {} |".format(
                    symbol,
                    reason,
                    int(summary["count"]),
                    int(summary["same_day_follow_up_opens"]),
                    int(summary["same_day_follow_up_closes"]),
                    float(summary["same_day_follow_up_pnl"]),
                    _format_counter_summary(summary["same_day_trigger_sources"]),
                    _format_counter_summary(summary["same_day_final_close_reasons"]),
                )
            )
    else:
        lines.append("| none | none | 0 | 0 | 0 | 0.00 | none | none |")

    return "\n".join(lines)


def _write_index(
    index_path: Path,
    version: str,
    date_range: str,
    timestamp: str,
    summary_listing: list[str],
    raw_listing: list[str],
) -> None:
    lines = [
        "# PropFirmPilot 生產環境日誌包",
        "",
        f"**版本**: {version}",
        f"**日期範圍**: {date_range}",
        f"**打包時間**: {timestamp}",
        "",
        "## 快速導覽",
        "",
        "### Bundle Metadata",
        "- bundle_manifest.json (machine-readable bundle context and included-file listing)",
        "",
        "### 摘要文件 (建議先讀)",
    ]
    lines.extend(
        summary_listing if summary_listing else ["- summary/ (no summary files generated)"]
    )
    lines.extend(["", "### 原始文件 (深挖用)"])
    lines.extend(raw_listing if raw_listing else ["- raw/ (no files collected)"])
    lines.extend(["", "## 已知問題", "{placeholder for user to fill}"])
    try:
        index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote INDEX {}", index_path)
    except OSError as exc:
        logger.error("Failed to write INDEX {}: {}", index_path, exc)


# ── Packaging ───────────────────────────────────────────────────────────────


def _zip_directory(src_dir: Path, output_zip: Path) -> None:
    try:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for path in src_dir.rglob("*"):
                if path.is_dir():
                    continue
                zipf.write(path, path.relative_to(src_dir))
        logger.info("Created zip archive {}", output_zip)
    except OSError as exc:
        logger.error("Failed to create zip {}: {}", output_zip, exc)


# ── Main Workflow ───────────────────────────────────────────────────────────


async def _run() -> None:
    args = _parse_args()
    version = _resolve_version(args.version)
    load_dotenv()

    config_path = Path(args.config)
    config = _load_merged_config(config_path)
    account_name = _resolve_account_name(config, config_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    date_range = f"{cutoff.date().isoformat()} to {datetime.now(timezone.utc).date().isoformat()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    folder_name = f"prod_logs_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{version}"

    output_base = Path(args.output_dir).resolve()
    output_dir = output_base / folder_name
    summary_dir = output_dir / "summary"
    raw_dir = output_dir / "raw"
    raw_logs_dir = raw_dir / "logs"
    raw_data_dir = raw_dir / "data"
    raw_config_dir = raw_dir / "config"

    _ensure_dir(summary_dir)
    _ensure_dir(raw_logs_dir)
    _ensure_dir(raw_data_dir)
    _ensure_dir(raw_config_dir)

    _write_config_snapshots(
        raw_config_dir=raw_config_dir,
        default_config_path=Path("config/default.yaml"),
        account_config_path=config_path,
        merged_config=config,
    )

    trade_journal_path = Path(
        _get_config_value(config, ["monitor", "trade_journal_path"], "data/trade_journal.jsonl")
    )
    memory_dir = Path(_get_config_value(config, ["monitor", "memory_dir"], "MEMORY"))
    decisions_db_path = Path(
        _get_config_value(config, ["decision_store", "db_path"], "data/decisions.db")
    )
    hwm_state_path = Path(
        _get_config_value(config, ["compliance", "hwm_state_path"], "data/hwm_state.json")
    )
    optimization_state_path = Path(
        _get_config_value(config, ["optimization", "state_path"], "data/optimization_state.json")
    )
    log_file_path = Path(_get_config_value(config, ["logging", "file"], "logs/prop_firm_pilot.log"))

    logger.info("Collecting logs from {}", log_file_path)
    _collect_logs(log_file_path, raw_logs_dir, cutoff)

    logger.info("Collecting data files")
    data_files = [
        trade_journal_path,
        decisions_db_path,
        hwm_state_path,
        optimization_state_path,
        Path("data/alpha158_fx_ic_ir_report.csv"),
    ]
    _collect_data_files(raw_data_dir, data_files, cutoff)

    logger.info("Collecting memory files from {}", memory_dir)
    raw_memory_dir = raw_dir / memory_dir.name
    memory_files = _collect_memory(memory_dir, raw_memory_dir, cutoff)

    logger.info("Collecting eval_results")
    _collect_eval_results(Path("eval_results"), raw_dir / "eval_results", cutoff)

    logger.info("Exporting Telegram messages")
    telegram_texts = await _export_telegram(raw_dir)

    log_content = _load_log_content(log_file_path, cutoff, MAX_LOG_CHARS)
    trade_content = _load_trade_journal(trade_journal_path, cutoff)
    decisions_dump = _dump_sqlite_db(decisions_db_path)
    memory_content = _load_memory_files(memory_files)
    telegram_content = "\n".join(telegram_texts)

    if args.no_summarize:
        logger.info("Skipping LLM summaries")
        _write_summary(
            summary_dir / "log_summary.md",
            _build_placeholder_summary(
                "Log Summary (Fallback)",
                "Summaries skipped (--no-summarize).",
                log_content,
            ),
        )
        _write_summary(
            summary_dir / "trades_summary.md",
            _build_placeholder_summary(
                "Trades Summary (Fallback)",
                "Summaries skipped (--no-summarize).",
                trade_content,
            ),
        )
        _write_summary(
            summary_dir / "decisions_summary.md",
            _build_decisions_fallback_summary(trade_content),
        )
        _write_summary(
            summary_dir / "memory_summary.md",
            _build_placeholder_summary(
                "Memory Summary (Fallback)",
                "Summaries skipped (--no-summarize).",
                memory_content,
            ),
        )
        _write_summary(
            summary_dir / "telegram_summary.md",
            _build_telegram_fallback_summary(telegram_content),
        )
    else:
        logger.info("Generating LLM summaries sequentially")
        await _summarize_all(
            summary_dir,
            log_content,
            trade_content,
            decisions_dump,
            memory_content,
            telegram_content,
        )

    summary_listing = _collect_summary_file_listing(summary_dir)
    raw_listing = _collect_raw_file_listing(raw_dir)
    _write_index(
        output_dir / "INDEX.md",
        version,
        date_range,
        timestamp,
        summary_listing,
        raw_listing,
    )
    _write_bundle_manifest(
        manifest_path=output_dir / "bundle_manifest.json",
        account_name=account_name,
        config_path=config_path.as_posix(),
        version=version,
        app_version=get_app_version(),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        days=args.days,
        date_range=date_range,
        bundle_folder=folder_name,
        zip_name=f"{folder_name}.zip",
        git_commit=_git_output(["git", "rev-parse", "HEAD"]),
        git_branch=_git_output(["git", "branch", "--show-current"]),
        included_logs=_collect_relative_paths(raw_logs_dir, output_dir),
        included_data_files=_collect_relative_paths(raw_data_dir, output_dir),
        included_config_files=_collect_relative_paths(raw_config_dir, output_dir),
        included_summary_files=_collect_relative_paths(summary_dir, output_dir),
    )

    output_zip = output_base / f"{folder_name}.zip"
    _zip_directory(output_dir, output_zip)
    remote_path = _upload_bundle_zip(output_zip, account_name)
    logger.info("Uploaded prod bundle to Dropbox: {}", remote_path)


def main() -> None:
    """CLI entry point."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
