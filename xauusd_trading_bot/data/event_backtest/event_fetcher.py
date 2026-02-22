"""
Event Fetcher - Fetches and stores all market events into the database.
Sources: FRED (economic), MT5 (DXY), GDELT (news), feedparser (RSS), holidays (calendar)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
import feedparser
import holidays as holidays_lib

from database import (
    initialize_database, upsert_economic_event, bulk_insert_dxy,
    upsert_news, upsert_holiday
)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ─── FRED Economic Data ───────────────────────────────────────────────────────

FRED_SERIES = {
    # Series ID : (event_name, direction_on_beat)
    # direction_on_beat: -1 = bearish gold when beat, +1 = bullish gold when beat
    'PAYEMS':       ('NFP',  -1.0),    # Nonfarm Payrolls - beat = hawkish = bearish gold
    'CPIAUCSL':     ('CPI',  -0.6),    # CPI - beat = inflation = mixed (higher rates hurts gold)
    'CPILFESL':     ('Core CPI', -0.7),# Core CPI
    'FEDFUNDS':     ('Fed Funds Rate', -0.8),  # Fed rate
    'UNRATE':       ('Unemployment', +0.7),    # Unemployment - beat (low) = bearish gold
    'GDPC1':        ('GDP', -0.5),     # GDP
}


def fetch_fred_data(start_date: datetime, end_date: datetime):
    """Fetch historical economic data from FRED API."""
    cache_file = CACHE_DIR / f"fred_{start_date.date()}_{end_date.date()}.json"

    if cache_file.exists():
        print("[FRED] Using cached data")
        with open(cache_file) as f:
            return json.load(f)

    try:
        import fredapi
        fred = fredapi.Fred()  # Uses FRED_API_KEY env var or no-auth (limited)
    except Exception:
        print("[FRED] fredapi not configured, using fallback")
        return []

    events = []
    for series_id, (name, direction) in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            prev_val = None
            for date, actual in data.items():
                if actual is None:
                    continue
                beat_miss = 'UNKNOWN'
                if prev_val is not None:
                    # Simple direction: compare to previous month
                    if direction < 0:
                        beat_miss = 'BEAT' if actual > prev_val else 'MISS'
                    else:
                        beat_miss = 'MISS' if actual > prev_val else 'BEAT'

                gold_bias = direction * (0.6 if beat_miss == 'BEAT' else -0.6 if beat_miss == 'MISS' else 0)

                events.append({
                    'event_date': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_name': name,
                    'currency': 'USD',
                    'impact': 'HIGH' if name in ('NFP', 'CPI', 'Core CPI', 'Fed Funds Rate') else 'MEDIUM',
                    'actual': float(actual),
                    'forecast': None,
                    'previous': float(prev_val) if prev_val else None,
                    'beat_miss': beat_miss,
                    'gold_bias': gold_bias,
                    'source': 'FRED'
                })
                prev_val = actual
            print(f"[FRED] {name}: {len([e for e in events if e['event_name'] == name])} data points")
            time.sleep(0.5)
        except Exception as ex:
            print(f"[FRED] Error fetching {series_id}: {ex}")

    with open(cache_file, 'w') as f:
        json.dump(events, f, indent=2)

    return events


# ─── FOMC Meeting Dates (Hardcoded - published a year in advance) ─────────────

FOMC_DECISIONS = [
    # Format: (date_str, 'HAWKISH'/'DOVISH'/'NEUTRAL', gold_bias, notes)
    ('2025-01-29', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-03-19', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-05-07', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-06-18', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-07-30', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-09-17', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-11-05', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2025-12-17', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2026-01-28', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
    ('2026-03-18', 'NEUTRAL', 0.0,   'Hold at 4.25-4.50%'),
]

# NFP Release Dates (1st Friday of each month, 13:30 UTC)
NFP_DATES = {
    '2025-11-07': {'event_name': 'NFP', 'actual': 227000, 'forecast': 223000, 'beat_miss': 'BEAT', 'gold_bias': -0.70},
    '2025-12-06': {'event_name': 'NFP', 'actual': 212000, 'forecast': 214000, 'beat_miss': 'MISS', 'gold_bias': +0.60},
    '2026-01-10': {'event_name': 'NFP', 'actual': 256000, 'forecast': 165000, 'beat_miss': 'BEAT', 'gold_bias': -0.85},
    '2026-02-07': {'event_name': 'NFP', 'actual': 143000, 'forecast': 170000, 'beat_miss': 'MISS', 'gold_bias': +0.65},
}

# CPI Release Dates
CPI_DATES = {
    '2025-11-13': {'event_name': 'CPI', 'actual': 2.6, 'forecast': 2.6, 'beat_miss': 'INLINE', 'gold_bias': 0.0},
    '2025-12-11': {'event_name': 'CPI', 'actual': 2.7, 'forecast': 2.7, 'beat_miss': 'INLINE', 'gold_bias': 0.0},
    '2026-01-15': {'event_name': 'CPI', 'actual': 2.9, 'forecast': 2.9, 'beat_miss': 'INLINE', 'gold_bias': 0.0},
    '2026-02-12': {'event_name': 'CPI', 'actual': 3.0, 'forecast': 2.9, 'beat_miss': 'BEAT',   'gold_bias': -0.50},
}


def fetch_hardcoded_events(start_date: datetime, end_date: datetime):
    """Use hardcoded known economic events for accuracy."""
    events = []

    # FOMC meetings
    for date_str, stance, bias, notes in FOMC_DECISIONS:
        dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=19, tzinfo=timezone.utc)
        if start_date <= dt <= end_date:
            events.append({
                'event_date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'event_name': f'FOMC {stance}',
                'currency': 'USD', 'impact': 'HIGH',
                'actual': None, 'forecast': None, 'previous': None,
                'beat_miss': stance, 'gold_bias': bias, 'source': 'hardcoded'
            })

    # NFP
    for date_str, data in NFP_DATES.items():
        dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=13, minute=30, tzinfo=timezone.utc)
        if start_date <= dt <= end_date:
            events.append({
                'event_date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'currency': 'USD', 'impact': 'HIGH',
                'actual': data['actual'], 'forecast': data['forecast'],
                'previous': None, 'source': 'hardcoded',
                **data
            })

    # CPI
    for date_str, data in CPI_DATES.items():
        dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=13, minute=30, tzinfo=timezone.utc)
        if start_date <= dt <= end_date:
            events.append({
                'event_date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'currency': 'USD', 'impact': 'HIGH',
                'actual': data['actual'], 'forecast': data['forecast'],
                'previous': None, 'source': 'hardcoded',
                **data
            })

    print(f"[Events] Hardcoded events loaded: {len(events)}")
    return events


# ─── DXY from MT5 ────────────────────────────────────────────────────────────

def fetch_dxy_history(start_date: datetime, end_date: datetime):
    """Fetch DXY_H6 (6-hour DXY bars) from MT5."""
    import MetaTrader5 as mt5
    mt5.initialize(r'C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe')
    mt5.login(52725397, server='ICMarketsSC-Demo')

    bars = mt5.copy_rates_range('DXY_H6', mt5.TIMEFRAME_H1, start_date, end_date)
    if bars is None or len(bars) == 0:
        print("[DXY] No H1 bars, trying H4...")
        bars = mt5.copy_rates_range('DXY_H6', mt5.TIMEFRAME_H4, start_date, end_date)
    mt5.shutdown()

    if bars is None or len(bars) == 0:
        print("[DXY] Could not fetch DXY data")
        return []

    records = []
    prev_close = None
    for b in bars:
        ts = datetime.fromtimestamp(b['time'], tz=timezone.utc).isoformat()
        change_pct = ((b['close'] - prev_close) / prev_close * 100) if prev_close else 0.0
        records.append((ts, b['open'], b['high'], b['low'], b['close'], change_pct))
        prev_close = b['close']

    print(f"[DXY] Fetched {len(records)} bars")
    return records


# ─── Holidays ─────────────────────────────────────────────────────────────────

def fetch_holidays(start_date: datetime, end_date: datetime):
    """Generate major market holidays affecting gold liquidity."""
    years = list(range(start_date.year, end_date.year + 1))
    holiday_records = []

    country_configs = [
        ('US',  holidays_lib.US(years=years),         0.6),  # US holiday = 60% liquidity
        ('CN',  holidays_lib.China(years=years),       0.4),  # China holiday = 40% liquidity
        ('GB',  holidays_lib.UnitedKingdom(years=years), 0.7),
        ('JP',  holidays_lib.Japan(years=years),       0.7),
    ]

    for country_code, holiday_obj, liquidity_impact in country_configs:
        for date, name in holiday_obj.items():
            if start_date.date() <= date <= end_date.date():
                holiday_records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'country': country_code,
                    'name': str(name),
                    'liquidity_impact': liquidity_impact
                })

    # Add Chinese New Year (Spring Festival) - extra low liquidity period
    cny_periods = [
        ('2026-01-28', '2026-02-05', 'Chinese New Year 2026'),
        ('2025-01-29', '2025-02-05', 'Chinese New Year 2025'),
    ]
    for cny_start, cny_end, name in cny_periods:
        dt = datetime.strptime(cny_start, '%Y-%m-%d')
        dt_end = datetime.strptime(cny_end, '%Y-%m-%d')
        while dt <= dt_end:
            if start_date.date() <= dt.date() <= end_date.date():
                holiday_records.append({
                    'date': dt.strftime('%Y-%m-%d'), 'country': 'CN',
                    'name': name, 'liquidity_impact': 0.3
                })
            dt += timedelta(days=1)

    print(f"[Holidays] Loaded {len(holiday_records)} holiday records")
    return holiday_records


# ─── News via GDELT ──────────────────────────────────────────────────────────

# Weighted keywords: (phrase, weight) — higher weight = stronger signal
GOLD_BULLISH_KEYWORDS = [
    # Strong bullish signals (weight 1.0)
    ('gold rally', 1.0), ('gold surge', 1.0), ('gold soars', 1.0), ('gold jumps', 1.0),
    ('safe haven demand', 1.0), ('rate cut', 0.8), ('dovish fed', 1.0),
    ('gold record', 0.9), ('gold all-time', 0.9), ('gold hits high', 1.0),
    # Medium bullish (weight 0.6)
    ('gold rise', 0.6), ('gold gain', 0.6), ('gold climb', 0.6), ('gold advance', 0.6),
    ('gold up', 0.5), ('gold higher', 0.5), ('gold edge', 0.4),
    ('safe haven', 0.7), ('recession fear', 0.7), ('inflation fear', 0.6),
    ('geopolitical', 0.5), ('risk off', 0.6), ('dollar weaken', 0.7),
    ('dollar fall', 0.6), ('dollar drop', 0.6), ('dollar slip', 0.5),
    ('gold demand', 0.6), ('gold buying', 0.6), ('gold inflow', 0.6),
    ('treasury yield fall', 0.6), ('treasury yield drop', 0.6), ('yield decline', 0.5),
    ('gold extend gain', 0.7), ('gold add', 0.4), ('bullion rise', 0.6),
    ('bullion gain', 0.6), ('precious metal rise', 0.5), ('precious metal gain', 0.5),
    ('gold rebound', 0.5), ('gold recover', 0.5), ('gold bounce', 0.5),
    ('rate cut hope', 0.7), ('rate cut bet', 0.6), ('rate cut expect', 0.6),
    ('fed pause', 0.4), ('fed dovish', 0.8), ('monetary easing', 0.6),
]
GOLD_BEARISH_KEYWORDS = [
    # Strong bearish signals (weight 1.0)
    ('gold crash', 1.0), ('gold selloff', 1.0), ('gold plunge', 1.0), ('gold tumble', 1.0),
    ('rate hike', 0.8), ('hawkish fed', 1.0), ('strong dollar', 0.8),
    # Medium bearish (weight 0.6)
    ('gold fall', 0.6), ('gold drop', 0.6), ('gold decline', 0.6), ('gold slip', 0.5),
    ('gold retreat', 0.6), ('gold lower', 0.5), ('gold down', 0.5),
    ('gold pressure', 0.6), ('gold extend loss', 0.7), ('gold lose', 0.5),
    ('gold dip', 0.4), ('gold ease', 0.4), ('gold shed', 0.5), ('gold weak', 0.5),
    ('risk on', 0.5), ('dollar surge', 0.7), ('dollar strength', 0.6),
    ('dollar rally', 0.6), ('dollar rise', 0.5), ('dollar gain', 0.5),
    ('dollar climb', 0.5), ('dollar high', 0.4), ('dollar jump', 0.6),
    ('treasury yield rise', 0.6), ('treasury yield jump', 0.6), ('yield surge', 0.5),
    ('gold outflow', 0.6), ('gold selling', 0.5), ('bullion fall', 0.6),
    ('bullion drop', 0.6), ('precious metal fall', 0.5), ('precious metal drop', 0.5),
    ('fed hawkish', 0.8), ('rate hike expect', 0.6), ('monetary tighten', 0.6),
    ('inflation hot', 0.5), ('inflation higher than', 0.5), ('cpi beat', 0.4),
]


def score_article_sentiment(title: str) -> float:
    """Score article sentiment for gold: -1.0 (bearish) to +1.0 (bullish).
    Two-step: 1) check phrase keywords, 2) word-level scoring for gold articles."""
    title_lower = title.lower()

    # Step 1: Exact phrase matching (high confidence)
    bullish_score = sum(w for kw, w in GOLD_BULLISH_KEYWORDS if kw in title_lower)
    bearish_score = sum(w for kw, w in GOLD_BEARISH_KEYWORDS if kw in title_lower)

    if bullish_score > 0 or bearish_score > 0:
        total = bullish_score + bearish_score
        raw = (bullish_score - bearish_score) / total
        return round(max(-1.0, min(1.0, raw)), 3)

    # Step 2: Word-level scoring for gold-related articles
    # Only if the article mentions gold/bullion
    words = set(title_lower.split())
    is_gold_article = bool(words & {'gold', 'bullion', 'xauusd', 'xau'})
    if not is_gold_article:
        return 0.0

    # Directional words (positive = bullish for gold)
    bullish_words = {
        'rally': 0.8, 'surge': 0.8, 'soar': 0.8, 'jump': 0.7, 'spike': 0.7,
        'rise': 0.5, 'rises': 0.5, 'rising': 0.5, 'gain': 0.5, 'gains': 0.5,
        'climb': 0.5, 'climbs': 0.5, 'advance': 0.5, 'advances': 0.5,
        'higher': 0.4, 'high': 0.3, 'highs': 0.4, 'record': 0.5,
        'rebound': 0.5, 'recover': 0.5, 'bounce': 0.4, 'upswing': 0.6,
        'best': 0.4, 'firm': 0.3, 'firms': 0.3, 'steady': 0.1,
        'inflow': 0.5, 'inflows': 0.5, 'demand': 0.4,
    }
    bearish_words = {
        'crash': 0.8, 'selloff': 0.8, 'plunge': 0.8, 'tumble': 0.7, 'plummet': 0.8,
        'fall': 0.5, 'falls': 0.5, 'falling': 0.5, 'drop': 0.5, 'drops': 0.5,
        'decline': 0.5, 'declines': 0.5, 'slip': 0.4, 'slips': 0.4,
        'lower': 0.4, 'low': 0.3, 'lows': 0.4, 'retreat': 0.5,
        'dip': 0.3, 'dips': 0.3, 'ease': 0.3, 'eases': 0.3,
        'shed': 0.4, 'weak': 0.4, 'weaken': 0.5, 'pressure': 0.4,
        'outflow': 0.5, 'outflows': 0.5, 'selling': 0.4,
        'loss': 0.4, 'losses': 0.4, 'lose': 0.4,
    }

    b_score = sum(bullish_words.get(w, 0) for w in words)
    s_score = sum(bearish_words.get(w, 0) for w in words)

    if b_score == 0 and s_score == 0:
        return 0.0

    total = b_score + s_score
    raw = (b_score - s_score) / total
    # Word-level is less reliable than phrase matching, scale down by 0.7
    return round(max(-1.0, min(1.0, raw * 0.7)), 3)


def fetch_gdelt_news(start_date: datetime, end_date: datetime):
    """Fetch historical gold news from GDELT using multiple queries for broad coverage."""
    cache_file = CACHE_DIR / f"gdelt_{start_date.date()}_{end_date.date()}.json"

    if cache_file.exists():
        print("[GDELT] Using cached data")
        with open(cache_file) as f:
            cached = json.load(f)
            if cached:  # Only use cache if non-empty
                return cached

    # Multiple query keywords for comprehensive gold news coverage
    queries = [
        "gold price",
        "gold rally safe haven",
        "gold falls drops",
        "federal reserve interest rate",
        "dollar strength weakness",
        "inflation CPI data",
    ]

    articles = []
    seen_urls = set()

    try:
        from gdeltdoc import GdeltDoc, Filters
        gd = GdeltDoc()

        for query in queries:
            print(f"[GDELT] Querying: '{query}'...")
            current = start_date
            while current < end_date:
                chunk_end = min(current + timedelta(days=30), end_date)
                try:
                    f = Filters(
                        keyword=query,
                        start_date=current.strftime('%Y-%m-%d'),
                        end_date=chunk_end.strftime('%Y-%m-%d'),
                    )
                    results = gd.article_search(f)
                    chunk_count = 0
                    if results is not None and not results.empty:
                        for _, row in results.iterrows():
                            url = str(row.get('url', ''))
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)

                            title = str(row.get('title', ''))
                            sentiment = score_article_sentiment(title)

                            # Parse seendate from GDELT format (YYYYMMDDTHHmmSSZ)
                            raw_date = str(row.get('seendate', ''))
                            published_at = _parse_gdelt_date(raw_date, current)

                            articles.append({
                                'published_at': published_at,
                                'title': title,
                                'source': str(row.get('domain', '')),
                                'url': url,
                                'sentiment': sentiment,
                                'gold_bias': sentiment * 0.6,
                                'keywords': [query]
                            })
                            chunk_count += 1
                        print(f"  {current.date()} - {chunk_end.date()}: +{chunk_count} new ({len(results)} total)")
                    else:
                        print(f"  {current.date()} - {chunk_end.date()}: 0 results")
                    time.sleep(2)  # Rate limit
                except Exception as ex:
                    print(f"  Chunk error {current.date()}: {ex}")
                    time.sleep(3)
                current = chunk_end + timedelta(days=1)
            time.sleep(1)

    except ImportError:
        print("[GDELT] gdeltdoc not installed, trying direct API...")
        articles = _fetch_gdelt_direct_api(start_date, end_date, queries, seen_urls)
    except Exception as ex:
        print(f"[GDELT] Library error: {ex}, trying direct API...")
        articles = _fetch_gdelt_direct_api(start_date, end_date, queries, seen_urls)

    # Only cache if we got actual results
    if articles:
        with open(cache_file, 'w') as f:
            json.dump(articles, f, indent=2)
        print(f"[GDELT] Total unique articles: {len(articles)}")
    else:
        print("[GDELT] No articles found - NOT caching empty result")

    return articles


def _parse_gdelt_date(raw_date: str, fallback: datetime) -> str:
    """Parse GDELT seendate format (YYYYMMDDTHHmmSSZ or similar) to ISO 8601."""
    if not raw_date:
        return fallback.isoformat()
    try:
        # GDELT format: "20260217T060000Z" or "2026-02-17T06:00:00Z"
        cleaned = raw_date.strip()
        if len(cleaned) >= 15 and cleaned[8] == 'T' and cleaned[:4].isdigit():
            # Already ISO-ish
            if '-' not in cleaned[:10]:
                # Convert YYYYMMDD to YYYY-MM-DD
                cleaned = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}T{cleaned[9:11]}:{cleaned[11:13]}:{cleaned[13:15]}+00:00"
            return cleaned
        # Try standard ISO parse
        dt = datetime.fromisoformat(cleaned.replace('Z', '+00:00'))
        return dt.isoformat()
    except Exception:
        return fallback.isoformat()


def _fetch_gdelt_direct_api(start_date: datetime, end_date: datetime,
                             queries: list, seen_urls: set) -> list:
    """Fallback: fetch from GDELT DOC API directly using requests."""
    import requests
    articles = []
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    for query in queries:
        print(f"[GDELT-API] Querying: '{query}'...")
        current = start_date
        while current < end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            try:
                params = {
                    'query': f'{query} sourcelang:eng',
                    'mode': 'artlist',
                    'maxrecords': 250,
                    'format': 'json',
                    'startdatetime': current.strftime('%Y%m%d%H%M%S'),
                    'enddatetime': chunk_end.strftime('%Y%m%d%H%M%S'),
                }
                resp = requests.get(base_url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    chunk_count = 0
                    for art in data.get('articles', []):
                        url = art.get('url', '')
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        title = art.get('title', '')
                        sentiment = score_article_sentiment(title)
                        published_at = _parse_gdelt_date(
                            art.get('seendate', ''), current)
                        articles.append({
                            'published_at': published_at,
                            'title': title,
                            'source': art.get('domain', ''),
                            'url': url,
                            'sentiment': sentiment,
                            'gold_bias': sentiment * 0.6,
                            'keywords': [query]
                        })
                        chunk_count += 1
                    print(f"  {current.date()} - {chunk_end.date()}: +{chunk_count} new")
                else:
                    print(f"  {current.date()}: HTTP {resp.status_code}")
                time.sleep(3)
            except Exception as ex:
                print(f"  Chunk error {current.date()}: {ex}")
                time.sleep(5)
            current = chunk_end + timedelta(days=1)

    return articles


def fetch_rss_news():
    """Fetch recent news from RSS feeds (for current/near-future use)."""
    feeds = [
        ('https://www.kitco.com/rss/gold.xml', 'Kitco'),
        ('https://www.fxstreet.com/rss/news', 'FXStreet'),
        ('https://www.marketwatch.com/rss/topstories', 'MarketWatch'),
    ]
    articles = []
    for url, source in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get('title', '')
                raw_published = entry.get('published', '')
                # Normalize RSS date (RFC 2822) → ISO 8601 for SQLite BETWEEN queries
                try:
                    if raw_published:
                        parsed_dt = parsedate_to_datetime(raw_published)
                        published = parsed_dt.astimezone(timezone.utc).isoformat()
                    else:
                        published = datetime.now(timezone.utc).isoformat()
                except Exception:
                    published = datetime.now(timezone.utc).isoformat()
                sentiment = score_article_sentiment(title)
                if abs(sentiment) > 0:  # Only store relevant articles
                    articles.append({
                        'published_at': published,
                        'title': title,
                        'source': source,
                        'url': entry.get('link', ''),
                        'sentiment': sentiment,
                        'gold_bias': sentiment * 0.6,
                        'keywords': []
                    })
            print(f"[RSS] {source}: {len(feed.entries)} entries")
        except Exception as ex:
            print(f"[RSS] {source} error: {ex}")
    return articles


# ─── Main Fetch & Store ───────────────────────────────────────────────────────

def fetch_and_store_all(start_date: datetime, end_date: datetime):
    """Fetch all event data for the backtest period and store in database."""
    print("\n" + "=" * 60)
    print("FETCHING ALL EVENT DATA")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)

    # 1. Initialize DB
    initialize_database()

    # 2. Hardcoded economic events (NFP, FOMC, CPI)
    print("\n[1/5] Loading economic events...")
    events = fetch_hardcoded_events(start_date, end_date)
    for event in events:
        upsert_economic_event(event)
    print(f"      Stored {len(events)} economic events")

    # 3. DXY from MT5
    print("\n[2/5] Fetching DXY from MT5...")
    dxy_records = fetch_dxy_history(start_date, end_date)
    if dxy_records:
        bulk_insert_dxy(dxy_records)
        print(f"      Stored {len(dxy_records)} DXY bars")

    # 4. Holidays
    print("\n[3/5] Loading market holidays...")
    holiday_records = fetch_holidays(start_date, end_date)
    for h in holiday_records:
        upsert_holiday(h)
    print(f"      Stored {len(holiday_records)} holiday records")

    # 5. GDELT historical news
    print("\n[4/5] Fetching GDELT historical news (this may take a while)...")
    gdelt_articles = fetch_gdelt_news(start_date, end_date)
    for article in gdelt_articles:
        upsert_news(article)
    print(f"      Stored {len(gdelt_articles)} GDELT articles")

    # 6. RSS current news
    print("\n[5/5] Fetching RSS feeds (current news)...")
    rss_articles = fetch_rss_news()
    for article in rss_articles:
        upsert_news(article)
    print(f"      Stored {len(rss_articles)} RSS articles")

    print("\n" + "=" * 60)
    print("FETCH COMPLETE - Database populated")
    print("=" * 60)


if __name__ == '__main__':
    from datetime import timezone
    start = datetime(2025, 11, 17, tzinfo=timezone.utc)
    end   = datetime(2026, 2, 17, tzinfo=timezone.utc)
    fetch_and_store_all(start, end)
