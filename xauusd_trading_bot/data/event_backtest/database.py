"""
Event Database - SQLite storage for all market events, news, and backtest results.
This is the bot's long-term memory for fundamental events.
"""

import sqlite3
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


def _normalize_to_iso(date_str: str) -> str:
    """Normalize any date format (RFC 2822, ISO 8601, etc.) to ISO 8601 UTC string.
    This ensures SQLite BETWEEN queries work correctly with lexicographic comparison."""
    if not date_str:
        return datetime.now(timezone.utc).isoformat()
    # Already ISO 8601? Starts with digits like "2026-02-17T..."
    if date_str[:4].isdigit():
        return date_str
    # Try RFC 2822 (e.g. "Tue, 17 Feb 2026 06:16:42 Z")
    try:
        parsed = parsedate_to_datetime(date_str)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    return date_str

DB_PATH = Path(__file__).parent / "db" / "events.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS economic_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date  TEXT NOT NULL,
            event_name  TEXT NOT NULL,
            currency    TEXT DEFAULT 'USD',
            impact      TEXT DEFAULT 'HIGH',
            actual      REAL,
            forecast    REAL,
            previous    REAL,
            beat_miss   TEXT,           -- 'BEAT', 'MISS', 'INLINE', 'UNKNOWN'
            gold_bias   REAL DEFAULT 0, -- -1.0 (bearish) to +1.0 (bullish)
            source      TEXT,
            fetched_at  TEXT,
            UNIQUE(event_date, event_name)
        );

        CREATE TABLE IF NOT EXISTS dxy_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL UNIQUE,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            change_pct  REAL            -- % change vs prev bar
        );

        CREATE TABLE IF NOT EXISTS news_articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT NOT NULL,
            title        TEXT,
            source       TEXT,
            url          TEXT UNIQUE,
            sentiment    REAL DEFAULT 0, -- -1.0 to +1.0
            keywords     TEXT,           -- JSON list
            gold_bias    REAL DEFAULT 0,
            stored_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS market_holidays (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            holiday_date     TEXT NOT NULL,
            country          TEXT NOT NULL,
            holiday_name     TEXT,
            liquidity_impact REAL DEFAULT 0.7, -- 0 = closed, 1 = full liquidity
            UNIQUE(holiday_date, country)
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date        TEXT,
            tier_label      TEXT,           -- 'baseline', 'tier_a', 'tier_b', 'all'
            tier_config     TEXT,           -- JSON of config used
            start_date      TEXT,
            end_date        TEXT,
            profit_factor   REAL,
            win_rate        REAL,
            total_return    REAL,
            max_drawdown    REAL,
            total_trades    INTEGER,
            avg_rr          REAL,
            score           REAL,
            optuna_trial    INTEGER DEFAULT -1
        );

        CREATE INDEX IF NOT EXISTS idx_events_date ON economic_events(event_date);
        CREATE INDEX IF NOT EXISTS idx_dxy_ts ON dxy_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_news_pub ON news_articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_holidays_date ON market_holidays(holiday_date);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {DB_PATH}")


# ─── Economic Events ──────────────────────────────────────────────────────────

def upsert_economic_event(event: dict):
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO economic_events
        (event_date, event_name, currency, impact, actual, forecast, previous, beat_miss, gold_bias, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event['event_date'], event['event_name'], event.get('currency', 'USD'),
        event.get('impact', 'HIGH'), event.get('actual'), event.get('forecast'),
        event.get('previous'), event.get('beat_miss', 'UNKNOWN'),
        event.get('gold_bias', 0.0), event.get('source', ''),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def get_events_in_window(timestamp: datetime, lookback_hours: int = 8, lookahead_hours: int = 1):
    """Get economic events within time window around a timestamp."""
    from datetime import timedelta
    start = (timestamp - timedelta(hours=lookback_hours)).isoformat()
    end   = (timestamp + timedelta(hours=lookahead_hours)).isoformat()

    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM economic_events
        WHERE event_date BETWEEN ? AND ?
        AND impact IN ('HIGH', 'MEDIUM')
        ORDER BY event_date
    """, (start, end)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── DXY History ──────────────────────────────────────────────────────────────

def bulk_insert_dxy(records: list):
    conn = get_connection()
    conn.executemany("""
        INSERT OR IGNORE INTO dxy_history (timestamp, open, high, low, close, change_pct)
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()


def get_dxy_at(timestamp: datetime):
    """Get DXY bar closest to (and before) the given timestamp."""
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM dxy_history
        WHERE timestamp <= ?
        ORDER BY timestamp DESC LIMIT 1
    """, (timestamp.isoformat(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_dxy_trend(timestamp: datetime, bars_back: int = 4):
    """Get DXY trend over last N bars (positive = USD strengthening = gold bearish)."""
    from datetime import timedelta
    start = (timestamp - timedelta(hours=bars_back * 6)).isoformat()
    conn = get_connection()
    rows = conn.execute("""
        SELECT close FROM dxy_history
        WHERE timestamp <= ? AND timestamp >= ?
        ORDER BY timestamp DESC LIMIT ?
    """, (timestamp.isoformat(), start, bars_back + 1)).fetchall()
    conn.close()
    if len(rows) < 2:
        return 0.0
    closes = [r['close'] for r in rows]
    return (closes[0] - closes[-1]) / closes[-1] * 100  # % change


# ─── News Articles ────────────────────────────────────────────────────────────

def upsert_news(article: dict):
    conn = get_connection()
    # Normalize date to ISO 8601 so SQLite BETWEEN queries work
    published_iso = _normalize_to_iso(article.get('published_at', ''))
    conn.execute("""
        INSERT OR IGNORE INTO news_articles
        (published_at, title, source, url, sentiment, keywords, gold_bias, stored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        published_iso, article.get('title', ''), article.get('source', ''),
        article.get('url', ''), article.get('sentiment', 0),
        json.dumps(article.get('keywords', [])), article.get('gold_bias', 0),
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()


def get_news_sentiment(timestamp: datetime, lookback_hours: int = 12):
    """Average gold sentiment from news in lookback window."""
    from datetime import timedelta
    start = (timestamp - timedelta(hours=lookback_hours)).isoformat()
    conn = get_connection()
    rows = conn.execute("""
        SELECT gold_bias FROM news_articles
        WHERE published_at BETWEEN ? AND ?
    """, (start, timestamp.isoformat())).fetchall()
    conn.close()
    if not rows:
        return 0.0
    return sum(r['gold_bias'] for r in rows) / len(rows)


# ─── Market Holidays ─────────────────────────────────────────────────────────

def upsert_holiday(holiday: dict):
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO market_holidays
        (holiday_date, country, holiday_name, liquidity_impact)
        VALUES (?, ?, ?, ?)
    """, (holiday['date'], holiday['country'], holiday['name'], holiday.get('liquidity_impact', 0.7)))
    conn.commit()
    conn.close()


def get_liquidity_factor(timestamp: datetime):
    """Return liquidity factor for a date (1.0 = normal, 0.5 = holiday)."""
    date_str = timestamp.strftime('%Y-%m-%d')
    conn = get_connection()
    rows = conn.execute("""
        SELECT liquidity_impact FROM market_holidays WHERE holiday_date = ?
    """, (date_str,)).fetchall()
    conn.close()
    if not rows:
        return 1.0
    return min(r['liquidity_impact'] for r in rows)  # Use most restrictive


# ─── Backtest Results ─────────────────────────────────────────────────────────

def save_backtest_run(run: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO backtest_runs
        (run_date, tier_label, tier_config, start_date, end_date,
         profit_factor, win_rate, total_return, max_drawdown, total_trades, avg_rr, score, optuna_trial)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(), run['tier_label'], json.dumps(run.get('tier_config', {})),
        run.get('start_date', ''), run.get('end_date', ''),
        run.get('profit_factor', 0), run.get('win_rate', 0),
        run.get('total_return', 0), run.get('max_drawdown', 0),
        run.get('total_trades', 0), run.get('avg_rr', 0),
        run.get('score', 0), run.get('optuna_trial', -1)
    ))
    conn.commit()
    conn.close()


def get_best_results():
    conn = get_connection()
    rows = conn.execute("""
        SELECT tier_label, profit_factor, win_rate, total_return, max_drawdown, total_trades, avg_rr, score
        FROM backtest_runs
        WHERE optuna_trial = -1
        ORDER BY score DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == '__main__':
    initialize_database()
    print("[DB] Database ready.")
