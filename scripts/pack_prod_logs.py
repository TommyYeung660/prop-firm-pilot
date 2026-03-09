"""
Pack production logs, data artifacts, and LLM summaries for PropFirmPilot.

Collects raw production outputs, exports Telegram messages, generates
LLM-optimized summaries via kimi-k2.5, builds an INDEX, and zips
everything into a single archive.

Usage:
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml --version v1.3.9
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml --version v1.3.9 --days 5
    python scripts/pack_prod_logs.py --config config/e8_one_5k_challenge.yaml --version v1.3.9 --no-summarize
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from dotenv import load_dotenv
from loguru import logger


# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
MODEL_NAME = "kimi-k2.5"
MAX_LOG_CHARS = 100_000


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
        required=True,
        help="Version string like v1.3.9",
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


def _collect_logs(log_file: Path, logs_dir: Path, cutoff: datetime) -> None:
    if log_file.exists():
        if _within_days(log_file, cutoff):
            _copy_file(log_file, logs_dir / log_file.name)
        else:
            _copy_file(log_file, logs_dir / log_file.name)
            logger.warning("Log file older than cutoff, still included: {}", log_file)
    else:
        logger.warning("Missing log file: {}", log_file)
    if log_file.parent.exists():
        for path in log_file.parent.glob("*.log*"):
            if _within_days(path, cutoff):
                _copy_file(path, logs_dir / path.name)


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
    """Call kimi-k2.5 via volcengine OpenAI-compatible API."""
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
    api_key = os.getenv("VOLCENGINE_API_KEY")
    if not api_key:
        logger.warning("VOLCENGINE_API_KEY not set; skipping LLM summaries")
        return

    async with httpx.AsyncClient() as client:
        try:
            log_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個生產環境日誌分析專家。分析以下 prop-firm-pilot 交易系統日誌，生成結構化摘要。",
                log_content,
            )
            _write_summary(summary_dir / "log_summary.md", log_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize logs: {}", exc)

        try:
            trades_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個交易績效分析專家。分析以下 JSONL 格式的交易日誌。",
                trade_content,
            )
            _write_summary(summary_dir / "trades_summary.md", trades_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize trades: {}", exc)

        try:
            decisions_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個 AI 決策系統分析專家。分析以下 SQLite 決策數據庫的 dump。",
                decisions_dump,
            )
            _write_summary(summary_dir / "decisions_summary.md", decisions_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize decisions: {}", exc)

        try:
            memory_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個交易記憶分析專家。分析以下每日 Markdown 交易記憶日誌。",
                memory_content,
            )
            _write_summary(summary_dir / "memory_summary.md", memory_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize memory: {}", exc)

        try:
            telegram_summary = await _call_llm(
                client,
                BASE_URL,
                api_key,
                "你是一個交易通知分析專家。分析以下 Telegram 交易通知歷史。",
                telegram_content,
            )
            _write_summary(summary_dir / "telegram_summary.md", telegram_summary)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to summarize Telegram: {}", exc)


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


def _write_index(
    index_path: Path,
    version: str,
    date_range: str,
    timestamp: str,
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
        "### 摘要文件 (建議先讀)",
        "| 文件 | 說明 |",
        "|------|------|",
        "| summary/log_summary.md | 主日誌摘要 — 錯誤/警告/關鍵事件 |",
        "| summary/trades_summary.md | 交易統計 — 勝率/PnL/每筆詳情 |",
        "| summary/decisions_summary.md | 決策分析 — BUY/SELL/HOLD 分佈 |",
        "| summary/memory_summary.md | 記憶日誌 — LLM 推理品質 |",
        "| summary/telegram_summary.md | Telegram 通知 — 系統事件 |",
        "",
        "### 原始文件 (深挖用)",
    ]
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
    load_dotenv()

    config_path = Path(args.config)
    config = _load_merged_config(config_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    date_range = f"{cutoff.date().isoformat()} to {datetime.now(timezone.utc).date().isoformat()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    folder_name = f"prod_logs_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{args.version}"

    output_base = Path(args.output_dir).resolve()
    output_dir = output_base / folder_name
    summary_dir = output_dir / "summary"
    raw_dir = output_dir / "raw"
    raw_logs_dir = raw_dir / "logs"
    raw_data_dir = raw_dir / "data"

    _ensure_dir(summary_dir)
    _ensure_dir(raw_logs_dir)
    _ensure_dir(raw_data_dir)

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

    log_content = _read_text_file(log_file_path, MAX_LOG_CHARS)
    trade_content = _load_trade_journal(trade_journal_path, cutoff)
    decisions_dump = _dump_sqlite_db(decisions_db_path)
    memory_content = _load_memory_files(memory_files)
    telegram_content = "\n".join(telegram_texts)

    if args.no_summarize:
        logger.info("Skipping LLM summaries")
        _write_summary(summary_dir / "log_summary.md", "Summaries skipped (--no-summarize).")
        _write_summary(summary_dir / "trades_summary.md", "Summaries skipped (--no-summarize).")
        _write_summary(summary_dir / "decisions_summary.md", "Summaries skipped (--no-summarize).")
        _write_summary(summary_dir / "memory_summary.md", "Summaries skipped (--no-summarize).")
        _write_summary(summary_dir / "telegram_summary.md", "Summaries skipped (--no-summarize).")
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

    raw_listing = _collect_raw_file_listing(raw_dir)
    _write_index(output_dir / "INDEX.md", args.version, date_range, timestamp, raw_listing)

    output_zip = output_base / f"{folder_name}.zip"
    _zip_directory(output_dir, output_zip)


def main() -> None:
    """CLI entry point."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
