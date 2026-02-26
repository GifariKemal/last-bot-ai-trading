"""
Smart Trader — Telegram Notifier
Notifications khusus untuk arsitektur LLM split (Python + Claude CLI).

Pesan yang dikirim:
  • BOT STARTED / STOPPED
  • SCAN REPORT  — heartbeat setiap 30 menit (hanya jika ada zona / skip)
  • ENTRY        — setelah Claude approve & order filled
  • EXIT         — SL/TP hit, scratch, claude take_profit
  • POSITION UPDATE — BE, Profit Lock, Trail, Stale Tighten, Claude Tighten

Gunakan sebagai singleton:
  import telegram_notifier as tg
  tg.init(token, chat_id, enabled)
  tg.get().send_entry(...)
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from loguru import logger

WIB = timezone(timedelta(hours=7))

# ── Session icons ─────────────────────────────────────────────────────────────
_SESSION_ICON = {
    "ASIAN":     "🌏",
    "LONDON":    "🇬🇧",
    "NEW_YORK":  "🇺🇸",
    "OVERLAP":   "🔥",
    "OFF_HOURS": "🌙",
}

# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional["SmartTraderNotifier"] = None


def init(token: str, chat_id: str, enabled: bool = True) -> None:
    global _instance
    _instance = SmartTraderNotifier(token, chat_id, enabled)


def get() -> Optional["SmartTraderNotifier"]:
    return _instance


# ── Helper ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    utc = datetime.now(timezone.utc)
    wib = utc.astimezone(WIB)
    return f"{utc.strftime('%d %b %Y %H:%M')} UTC / {wib.strftime('%H:%M')} WIB"


def _ts_short() -> str:
    utc = datetime.now(timezone.utc)
    wib = utc.astimezone(WIB)
    return f"{utc.strftime('%H:%M')} UTC / {wib.strftime('%H:%M')} WIB"


def _pnl_emoji(pnl: float) -> str:
    if pnl > 0.5:
        return "🏆"
    if pnl < -0.5:
        return "❌"
    return "⚖️"


def _pnl_label(pnl: float) -> str:
    if pnl > 0.5:
        return "WIN"
    if pnl < -0.5:
        return "LOSS"
    return "SCRATCH"


def _dir_emoji(direction: str) -> str:
    return "🟢" if direction == "LONG" else "🔴"


def _ema_emoji(ema: str) -> str:
    return {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➖"}.get(ema, "➖")


def _conf_bar(conf: float) -> str:
    """Visual confidence bar  e.g. 0.82 → ████░ 8.2/10"""
    filled = round(conf * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {conf * 10:.1f}/10"


# ── Main class ────────────────────────────────────────────────────────────────

class SmartTraderNotifier:

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self.token    = token
        self.chat_id  = str(chat_id)
        self.enabled  = enabled
        self.base_url = f"https://api.telegram.org/bot{token}"

    # ── Core sender (non-blocking daemon thread) ──────────────────────────────

    def _send(self, text: str) -> bool:
        if not self.enabled or not self.token:
            return False

        def _do():
            try:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id":                  self.chat_id,
                        "text":                     text,
                        "parse_mode":               "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if not resp.ok:
                    logger.warning(f"Telegram send failed: {resp.status_code} {resp.text[:150]}")
            except Exception as e:
                logger.warning(f"Telegram error: {e}")

        threading.Thread(target=_do, daemon=True).start()
        return True

    # ── BOT STARTED ───────────────────────────────────────────────────────────

    def send_bot_started(
        self,
        balance: float,
        equity: float,
        login: int,
        server: str,
        symbol: str,
        leverage: int,
        model: str,
        sl_mult: float,
        tp_mult: float,
        max_pos: int,
        min_signals: int,
        min_conf: float,
    ) -> bool:
        try:
            eq_delta = equity - balance
            eq_em    = "🟢" if eq_delta >= 0 else "🔴"
            rr_est   = tp_mult / sl_mult

            L = [
                "🤖 <b>SMART TRADER — STARTED</b>",
                f"Model: <b>{model}</b> | {symbol} @ {server}",
                "",
                "[ KONFIGURASI ]",
                f"› SL: {sl_mult:.1f}× ATR | TP: {tp_mult:.1f}× ATR | RR ~1:{rr_est:.1f}",
                f"› Min Sinyal: {min_signals} | Min Conf: {min_conf:.0%}",
                f"› Max Posisi: {max_pos} | Exit: BE(0.7R) → Lock(1.5R) → Trail(40%)",
                "",
                "[ AKUN ]",
                f"💰 Balance: <b>${balance:,.2f}</b> | "
                f"Equity: {eq_em} ${equity:,.2f} ({eq_delta:+.2f})",
                f"› Login: {login} | Leverage: 1:{leverage}",
                "",
                f"⏰ {_ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram bot_started error: {e}")
            return False

    # ── BOT STOPPED ───────────────────────────────────────────────────────────

    def send_bot_stopped(
        self,
        reason: str = "Manual stop",
        session_trades: int = 0,
        session_pnl: float = 0.0,
    ) -> bool:
        try:
            pnl_em = "🟢" if session_pnl >= 0 else "🔴"
            L = [
                "🛑 <b>SMART TRADER — STOPPED</b>",
                f"› Alasan: {reason}",
            ]
            if session_trades > 0:
                L.append(
                    f"› Sesi ini: {session_trades} trade | "
                    f"{pnl_em} ${session_pnl:+.2f}"
                )
            L += ["", f"⏰ {_ts()}"]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram bot_stopped error: {e}")
            return False

    # ── SCAN REPORT (heartbeat 30 menit) ──────────────────────────────────────

    def send_scan_report(
        self,
        price: float,
        spread: float,
        rsi: float,
        atr: float,
        ema_trend: str,
        pd_zone: str,
        session: str,
        zones_total: int,
        nearby_zones: list[dict],
        skip_reason: str = "",
        balance: float = 0,
        equity: float = 0,
        open_positions: int = 0,
    ) -> bool:
        """
        Heartbeat scan report. Dikirim setiap 30 menit.
        Tampilkan kondisi pasar + zona terdekat + alasan skip jika ada.
        """
        try:
            flag = _SESSION_ICON.get(session, "")
            ema_em = _ema_emoji(ema_trend)

            # Price movement indicator
            rsi_tag = ""
            if rsi > 70:
                rsi_tag = " ⚠️OB"
            elif rsi < 30:
                rsi_tag = " ⚠️OS"

            # Nearby zone summary
            zone_lines = []
            for z in nearby_zones[:3]:
                ztype  = z.get("type", "?")
                zdist  = z.get("distance_pts", 0)
                zem    = "🟢" if "BULL" in ztype or "BOS_BULL" in ztype else "🔴"
                zone_lines.append(f"  {zem} {ztype} ({zdist:.1f}pt away)")

            L = [
                f"📡 <b>XAUUSD — {session}</b> {flag}",
                f"💹 {price:.2f} | Spread: {spread:.1f} | RSI: {rsi:.0f}{rsi_tag} | ATR: {atr:.1f}",
                "",
                "[ MARKET ]",
                f"› EMA(50): <b>{ema_trend}</b> {ema_em} | P/D: {pd_zone}",
                f"› Zona: {zones_total} terdeteksi | {len(nearby_zones)} nearby",
            ]

            if zone_lines:
                L.append("[ ZONA TERDEKAT ]")
                L += zone_lines

            # Status
            L.append("")
            L.append("[ STATUS ]")
            if open_positions > 0:
                L.append(f"🔄 {open_positions} posisi aktif")
            if skip_reason:
                L.append(f"⏸ WATCHING — {skip_reason}")
            elif nearby_zones:
                L.append(f"👀 ZONA DEKAT — menunggu sinyal cukup")
            else:
                L.append("⏳ Scanning...")

            if balance > 0:
                eq_delta = equity - balance
                eq_em = "🟢" if eq_delta >= 0 else "🔴"
                L.append(f"💰 ${balance:.2f} | Equity: {eq_em} ${equity:.2f} ({eq_delta:+.2f})")

            L += ["", f"⏰ {_ts_short()}"]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram scan_report error: {e}")
            return False

    # ── ENTRY ─────────────────────────────────────────────────────────────────

    def send_entry(
        self,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        lot: float,
        ticket: int,
        zone_type: str,
        zone_dist: float,
        signals: list[str],
        signal_count: int,
        confidence: float,
        claude_reason: str,
        session: str,
        ema_trend: str,
        rsi: float,
        atr: float,
        claude_latency_ms: float = 0,
        claude_tokens: int = 0,
        # Enriched fields (hybrid architecture)
        regime: str = "",
        pd_zone: str = "",
        pre_score: float = 0.0,
        tier1_count: int = 0,
        tier2_count: int = 0,
        has_structure: bool = False,
        ema_aligned: bool = False,
        pd_aligned: bool = False,
    ) -> bool:
        try:
            dir_em  = _dir_emoji(direction)
            flag    = _SESSION_ICON.get(session, "")
            ema_em  = _ema_emoji(ema_trend)
            sl_dist = abs(price - sl)
            tp_dist = abs(tp - price)
            rr      = tp_dist / sl_dist if sl_dist > 0 else 0
            signals_str = " + ".join(signals) if signals else "—"
            review_str = f"› Review: {claude_latency_ms / 1000:.1f}s | ~{claude_tokens} tokens" \
                         if claude_latency_ms > 0 else ""

            # Signal tier breakdown
            struct_tag = "BOS/CHoCH" if has_structure else "no structure"
            ema_tag = "aligned" if ema_aligned else "counter" if not ema_aligned and ema_trend != "NEUTRAL" else "neutral"
            pd_tag = "aligned" if pd_aligned else pd_zone

            L = [
                f"{dir_em} <b>ENTRY — {direction}</b> @ {price:.2f}",
                "",
                "[ SETUP ]",
                f"🎯 Entry: {price:.2f}",
                f"🛑 SL: {sl:.2f}  (-{sl_dist:.1f}pt)",
                f"💎 TP: {tp:.2f}  (+{tp_dist:.1f}pt)",
                f"📊 R:R 1:{rr:.1f} | Lot: {lot:.2f}",
                "",
                "[ CLAUDE AI ]",
                f"› Confidence: {_conf_bar(confidence)}",
                f"› Sinyal ({signal_count}): {signals_str}",
                f"› Tier-1: {tier1_count} | Tier-2: {tier2_count} | {struct_tag}",
                f"› Zone: {zone_type} ({zone_dist:.1f}pt away)",
                f'› Reason: "<i>{claude_reason}</i>"',
            ]
            if review_str:
                L.append(review_str)
            L += [
                "",
                "[ KONTEKS ]",
                f"› Session: {session} {flag} | EMA: {ema_trend} {ema_em} ({ema_tag})",
                f"› RSI: {rsi:.0f} | ATR: {atr:.1f} | P/D: {pd_tag}",
                f"› Regime: {regime} | Pre-score: {pre_score:.2f}",
                "",
                f"⏰ {_ts()} | #{ticket}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram entry error: {e}")
            return False

    # ── EXIT ──────────────────────────────────────────────────────────────────

    def send_exit(
        self,
        direction: str,
        ticket: int,
        entry_price: float,
        exit_price: float,
        pnl_pts: float,
        pnl_usd: float,
        age_min: float,
        reason: str,
    ) -> bool:
        try:
            emoji  = _pnl_emoji(pnl_usd)
            label  = _pnl_label(pnl_usd)
            dir_em = _dir_emoji(direction)

            # Duration string
            if age_min < 60:
                dur_str = f"{age_min:.0f}min"
            else:
                h = int(age_min // 60)
                m = int(age_min % 60)
                dur_str = f"{h}h{m:02d}m"

            # Arrow direction of price move
            move_em = "🟢" if pnl_pts > 0 else "🔴"

            # Reason tag
            reason_map = {
                "scratch_exit":       "⚖️ Scratch (flat)",
                "claude_take_profit": "🧠 Claude Take Profit",
                "tp_hit":             "💎 TP Hit",
                "sl_hit":             "🛑 SL Hit",
            }
            reason_str = reason_map.get(reason, reason)

            L = [
                f"{emoji} <b>{label} {pnl_usd:+.2f} USD</b> | {direction} #{ticket}",
                f"› {dir_em} {entry_price:.2f} → {move_em} {exit_price:.2f} "
                f"({pnl_pts:+.1f}pt)",
                f"› ⏱ {dur_str} | {reason_str}",
                "",
                f"⏰ {_ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram exit error: {e}")
            return False

    # ── POSITION CLOSED EXTERNALLY (SL/TP hit by broker) ────────────────────────

    def send_position_closed_external(
        self,
        ticket: int,
        direction: str,
        entry_price: float,
        last_sl: float,
        last_tp: float,
        close_price: float,
        pnl_pts: float,
        pnl_usd: float,
        age_min: float,
        close_type: str,
    ) -> bool:
        """Notify when a position disappears (closed by broker via SL/TP hit)."""
        try:
            emoji  = _pnl_emoji(pnl_usd)
            label  = _pnl_label(pnl_usd)
            dir_em = _dir_emoji(direction)
            move_em = "\U0001f7e2" if pnl_pts > 0 else "\U0001f534"

            if age_min < 60:
                dur_str = f"{age_min:.0f}min"
            else:
                h = int(age_min // 60)
                m = int(age_min % 60)
                dur_str = f"{h}h{m:02d}m"

            close_map = {
                "sl_hit":    "\U0001f6d1 Trailing SL Hit",
                "tp_hit":    "\U0001f48e TP Hit",
                "unknown":   "\u2753 Closed Externally",
            }
            close_str = close_map.get(close_type, close_type)

            L = [
                f"{emoji} <b>{label} {pnl_usd:+.2f} USD</b> | {direction} #{ticket}",
                f"\u203a {dir_em} {entry_price:.2f} \u2192 {move_em} {close_price:.2f} "
                f"({pnl_pts:+.1f}pt)",
                f"\u203a \u23f1 {dur_str} | {close_str}",
                f"\u203a SL: {last_sl:.2f} | TP: {last_tp:.2f}",
                "",
                f"\u23f0 {_ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram closed_external error: {e}")
            return False

    # ── POSITION UPDATE (BE / Lock / Trail / Stale / Claude Tighten) ──────────

    def send_position_update(
        self,
        ticket: int,
        direction: str,
        action: str,
        old_sl: float,
        new_sl: float,
        pnl_pts: float,
        extra: str = "",
    ) -> bool:
        """
        action: "BE" | "PROFIT_LOCK" | "TRAIL" | "STALE_TIGHTEN" | "CLAUDE_TIGHTEN"
        """
        try:
            action_icon = {
                "BE":             "⚡ BE",
                "PROFIT_LOCK":    "🔒 PROFIT LOCK",
                "TRAIL":          "🎯 TRAIL",
                "STALE_TIGHTEN":  "⏳ STALE TIGHTEN",
                "CLAUDE_TIGHTEN": "🧠 CLAUDE TIGHTEN",
            }.get(action, f"⚙️ {action}")

            dir_em = _dir_emoji(direction)
            L = [
                f"{action_icon} | {direction} #{ticket}",
                f"› {dir_em} SL: {old_sl:.2f} → {new_sl:.2f}",
                f"› P/L saat ini: {pnl_pts:+.1f}pt",
            ]
            if extra:
                L.append(f"› {extra}")
            L.append(f"⏰ {_ts_short()}")
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram position_update error: {e}")
            return False

    # ── CLAUDE EXIT REVIEW RESULT ─────────────────────────────────────────────

    def send_claude_exit_review(
        self,
        ticket: int,
        direction: str,
        action: str,
        pnl_pts: float,
        pnl_usd: float,
        claude_reason: str,
        new_sl: float = 0,
        old_sl: float = 0,
        claude_latency_ms: float = 0,
        claude_tokens: int = 0,
    ) -> bool:
        """
        Kirim hasil exit review Claude: HOLD / TAKE_PROFIT / TIGHTEN.
        ALL actions notified (including HOLD — compact format).
        """
        try:
            metrics_str = ""
            if claude_latency_ms > 0:
                metrics_str = f" | {claude_latency_ms / 1000:.1f}s | ~{claude_tokens}tok"

            if action == "HOLD":
                # Compact 1-2 line format to avoid spam
                dir_em = _dir_emoji(direction)
                L = [
                    f"🧠 HOLD | {dir_em} {direction} #{ticket} | "
                    f"P/L: {pnl_pts:+.1f}pt | {claude_reason}{metrics_str}",
                ]
                return self._send("\n".join(L))

            if action == "TAKE_PROFIT":
                icon  = "🧠💰 <b>CLAUDE TAKE PROFIT</b>"
                lines = [
                    f"› {direction} #{ticket} | P/L: {pnl_pts:+.1f}pt (${pnl_usd:+.2f})",
                    f'› Reason: "<i>{claude_reason}</i>"',
                ]
            else:  # TIGHTEN
                icon  = "🧠🔧 <b>CLAUDE TIGHTEN</b>"
                lines = [
                    f"› {direction} #{ticket} | P/L: {pnl_pts:+.1f}pt",
                    f"› SL: {old_sl:.2f} → {new_sl:.2f}",
                    f'› Reason: "<i>{claude_reason}</i>"',
                ]

            if metrics_str:
                lines.append(f"› Review:{metrics_str}")

            L = [icon] + lines + ["", f"⏰ {_ts_short()}"]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram claude_exit error: {e}")
            return False

    # ── ZONE ALERT ────────────────────────────────────────────────────────────

    def send_zone_alert(
        self,
        direction: str,
        zone_type: str,
        zone_dist: float,
        signal_count: int,
        signals: list[str],
        skip_reason: str,
        price: float,
        session: str,
    ) -> bool:
        """
        Zona terdeteksi tapi entry diblok — kirim alert sekali per zona.
        Berguna untuk tracking sinyal yang dilewatkan.
        """
        try:
            dir_em  = _dir_emoji(direction)
            flag    = _SESSION_ICON.get(session, "")
            sig_str = " + ".join(signals) if signals else "—"

            L = [
                f"👀 <b>ZONA TERDETEKSI</b> — diblok",
                f"› {dir_em} {direction} | {zone_type} ({zone_dist:.1f}pt away)",
                f"› Sinyal: {signal_count} [{sig_str}]",
                f"› Alasan skip: <i>{skip_reason}</i>",
                f"› {price:.2f} | {session} {flag}",
                "",
                f"⏰ {_ts_short()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram zone_alert error: {e}")
            return False

    # ── HOURLY ANALYSIS REPORT ──────────────────────────────────────────────

    def send_hourly_report(
        self,
        price: float,
        spread: float,
        rsi: float,
        atr: float,
        ema_trend: str,
        pd_zone: str,
        h4_bias: str,
        session: str,
        zones_total: int,
        nearby_zones: list[dict],
        nearest_zone: dict | None,
        proximity: float,
        balance: float = 0,
        equity: float = 0,
        open_positions: list[dict] | None = None,
        session_trades: int = 0,
    ) -> bool:
        """Compact hourly analysis report."""
        try:
            flag = _SESSION_ICON.get(session, "")

            # Outlook based on H4 + EMA alignment
            h4_em = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGING": "🟡"}.get(h4_bias, "⚪")
            ema_aligned = (
                (h4_bias == "BULLISH" and ema_trend == "BULLISH") or
                (h4_bias == "BEARISH" and ema_trend == "BEARISH")
            )
            align_tag = "aligned" if ema_aligned else "divergent"

            # RSI tag
            if rsi > 80:
                rsi_tag = " OB"
            elif rsi > 70:
                rsi_tag = " OB"
            elif rsi < 20:
                rsi_tag = " OS"
            elif rsi < 30:
                rsi_tag = " OS"
            else:
                rsi_tag = ""

            # P/D tag
            pd_em = {"PREMIUM": "🔴", "DISCOUNT": "🟢", "EQUILIBRIUM": "⚪"}.get(pd_zone, "")

            L = [
                f"🤖 <b>Smart Trader — {session}</b> {flag}",
                f"Outlook: {h4_bias} {h4_em} | EMA: {ema_trend} ({align_tag})",
                "",
                "[ MARKET ]",
                f"💹 <b>{price:.2f}</b> | RSI {rsi:.0f}{rsi_tag} | ATR {atr:.1f}",
                f"└ EMA: {ema_trend} | P/D: {pd_zone} {pd_em} | Spread: {spread:.1f}",
            ]

            # Zone section
            L += ["", "[ ZONA SMC ]"]
            L.append(f"› Total: {zones_total} | Nearby: {len(nearby_zones)} (&lt; {proximity}pt)")

            if nearby_zones:
                for z in nearby_zones[:3]:
                    ztype = z.get("type", "?")
                    zdist = z.get("distance_pts", 0)
                    zem = "🟢" if "BULL" in ztype else "🔴"
                    L.append(f"› {zem} {ztype} ({zdist:.1f}pt) — READY")
            elif nearest_zone:
                nz_type = nearest_zone.get("type", "?")
                nz_dist = nearest_zone.get("distance_pts", 0)
                nz_level = nearest_zone.get("high") or nearest_zone.get("level", 0)
                nz_em = "🟢" if "BULL" in nz_type else "🔴"
                arrow = "↓" if price > nz_level else "↑"
                L.append(f"› Terdekat: {nz_em} {nz_type} @ {nz_level:.0f} ({nz_dist:.0f}pt {arrow})")
            else:
                L.append("› Tidak ada zona — menunggu struktur baru")

            # Position section
            L += ["", "[ POSISI ]"]
            if open_positions:
                for p in open_positions:
                    dir_em = "🟢" if p["direction"] == "LONG" else "🔴"
                    pnl = p.get("pnl", 0)
                    pnl_sign = "+" if pnl >= 0 else ""
                    L.append(
                        f"› {dir_em} {p['direction']} #{p['ticket']} | "
                        f"Entry {p['entry']:.2f} → {p['current']:.2f}"
                    )
                    L.append(f"  P/L: <b>${pnl_sign}{pnl:.2f}</b> | SL {p['sl']:.2f} | TP {p['tp']:.2f}")
            else:
                L.append("› Kosong — belum ada entry")

            # Account
            if balance > 0:
                eq_delta = equity - balance
                eq_em = "🟢" if eq_delta >= 0 else "🔴"
                L += ["", "[ AKUN ]"]
                L.append(f"💰 ${balance:,.2f} | Eq: {eq_em} ${equity:,.2f} ({eq_delta:+.2f})")
                L.append(f"› Trades sesi: {session_trades}")

            # Verdict / outlook
            L += ["", "[ VERDICT ]"]
            if nearby_zones:
                L.append("✅ Zona dekat — menunggu sinyal + Claude")
            elif nearest_zone and nearest_zone.get("distance_pts", 999) <= 50:
                nd = nearest_zone.get("distance_pts", 0)
                L.append(f"⏳ Zona {nd:.0f}pt away — menunggu harga mendekati")
            elif zones_total > 0:
                L.append("⏳ Zona jauh — ATH territory, menunggu retrace")
            else:
                L.append("⏸ No structure — menunggu H1 BOS/FVG/OB")

            L += ["", f"⏰ {_ts()}"]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram hourly_report error: {e}")
            return False

    # ── CHART IMAGE ────────────────────────────────────────────────────────────

    def send_chart(self, image_path: str, caption: str = "") -> bool:
        """Send a chart image (PNG) to Telegram via sendPhoto."""
        if not self.enabled or not self.token:
            return False

        def _do():
            try:
                with open(image_path, "rb") as photo:
                    data = {
                        "chat_id":    self.chat_id,
                        "parse_mode": "HTML",
                    }
                    if caption:
                        data["caption"] = caption[:1024]  # Telegram caption limit
                    resp = requests.post(
                        f"{self.base_url}/sendPhoto",
                        data=data,
                        files={"photo": ("chart.png", photo, "image/png")},
                        timeout=30,
                    )
                    if not resp.ok:
                        logger.warning(f"Telegram sendPhoto failed: {resp.status_code} {resp.text[:150]}")
                    else:
                        logger.info(f"Chart sent to Telegram: {image_path}")
            except Exception as e:
                logger.warning(f"Telegram sendPhoto error: {e}")

        threading.Thread(target=_do, daemon=True).start()
        return True

    # ── ERROR / WARNING ───────────────────────────────────────────────────────

    def send_error(self, title: str, detail: str = "") -> bool:
        try:
            L = [f"⚠️ <b>{title}</b>"]
            if detail:
                L.append(f"› {detail}")
            L += ["", f"⏰ {_ts_short()}"]
            return self._send("\n".join(L))
        except Exception as e:
            logger.warning(f"Telegram error notification failed: {e}")
            return False
