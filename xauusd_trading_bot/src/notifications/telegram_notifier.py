"""
Telegram Notifier
Professional structured notifications with section-based layout.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from ..bot_logger import get_logger

WIB = timezone(timedelta(hours=7))

SESSION_FLAGS = {
    "Asian Session": "\U0001f1ef\U0001f1f5",
    "London Session": "\U0001f1ec\U0001f1e7",
    "New York Session": "\U0001f1fa\U0001f1f8",
    "London-NY Overlap": "\U0001f1ec\U0001f1e7\U0001f1fa\U0001f1f8",
    "Off-Session": "\U0001f319",
    "No Session": "\U0001f319",
}


class TelegramNotifier:
    """Send structured trading notifications to Telegram."""

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self.logger = get_logger()
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._prev_candle: Optional[Dict] = None

    # ─── Core Sender ─────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        if not self.enabled:
            return False

        def _do_send():
            try:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if not resp.ok:
                    self.logger.warning(
                        f"Telegram send failed: {resp.status_code} {resp.text[:200]}"
                    )
            except Exception as e:
                self.logger.warning(f"Telegram error: {e}")

        threading.Thread(target=_do_send, daemon=True).start()
        return True

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        utc = datetime.now(timezone.utc)
        wib = datetime.now(WIB)
        return f"{utc.strftime('%d %b %Y %H:%M')} UTC / {wib.strftime('%H:%M')} WIB"

    @staticmethod
    def _ts_short() -> str:
        utc = datetime.now(timezone.utc)
        wib = datetime.now(WIB)
        return f"{utc.strftime('%H:%M')} UTC / {wib.strftime('%H:%M')} WIB"

    @staticmethod
    def _enum_str(val) -> str:
        return val.value if hasattr(val, "value") else str(val)

    @staticmethod
    def _rating(score: float, threshold: float) -> str:
        margin = score - threshold
        if margin >= 0.20:
            return "A+"
        if margin >= 0.10:
            return "A"
        if margin >= 0.0:
            return "B"
        if margin >= -0.10:
            return "C"
        return "D"

    @staticmethod
    def _session_flag(name: str) -> str:
        return SESSION_FLAGS.get(name, "")

    @staticmethod
    def _outlook(best_dir: str, best_score: float) -> tuple:
        if best_score < 0.30:
            return "NEUTRAL", "\u2796"
        if best_dir == "LONG":
            return "BULLISH", "\U0001f7e2"
        return "BEARISH", "\U0001f534"

    @staticmethod
    def _pnl_emoji(profit: float) -> str:
        if profit > 0.5:
            return "\U0001f3c6"
        elif profit < -0.5:
            return "\u274c"
        return "\u2696\ufe0f"

    @staticmethod
    def _pnl_label(profit: float) -> str:
        if profit > 0.5:
            return "WIN"
        elif profit < -0.5:
            return "LOSS"
        return "BE"

    @staticmethod
    def _smc_scores(conf: Dict) -> Dict[str, float]:
        brk = conf.get("breakdown", {})
        details = brk.get("smc", {}).get("details", {})
        return {
            "choch": details.get("choch", 0),
            "bos": details.get("bos", 0),
            "ob": details.get("order_block", 0),
            "fvg": details.get("fvg", 0),
            "liq": details.get("liquidity_sweep", 0),
        }

    # ─── Signal Analysis (every 15 min) ─────────────────────────

    def send_signal_analysis(self, market_data: Dict,
                             gate_result: Optional[Dict] = None,
                             position_info: Optional[Dict] = None) -> bool:
        """Send structured 15-minute candle analysis with position & gate info."""
        try:
            price = market_data["current_price"]
            tech = market_data["technical_indicators"]
            smc = market_data["smc_analysis"]
            confluence = market_data["confluence_scores"]
            market = market_data["market_analysis"]
            trend = market_data["trend_analysis"]
            mtf = market_data["mtf_analysis"]
            vol = market_data.get("volatility_analysis", {})

            session_name = market_data.get("session_name", "Unknown")
            threshold = market_data.get("threshold", 0.55)
            regime = market_data.get("regime", "unknown")

            atr = tech["atr"]
            rsi = tech["rsi"]
            trend_str = self._enum_str(trend.get("direction", ""))
            vol_str = self._enum_str(vol.get("level", ""))
            ema_20 = tech.get("ema_20", 0)
            ema_50 = tech.get("ema", {}).get(50, 0)
            macd_h = tech.get("macd", {}).get("histogram")
            h1_bias = mtf.get("h1_bias", "neutral")
            mtf_aligned = mtf.get("is_aligned", False)

            # Confluence
            bull_c = confluence["bullish"]
            bear_c = confluence["bearish"]
            bull_score = bull_c["score"]
            bear_score = bear_c["score"]
            best_score = max(bull_score, bear_score)
            best_dir = "LONG" if bull_score >= bear_score else "SHORT"
            dom_key = "bullish" if best_dir == "LONG" else "bearish"
            dom_smc = smc[dom_key]

            has_signal = best_score >= threshold
            confidence = best_score * 10
            outlook_label, outlook_emoji = self._outlook(best_dir, best_score)
            flag = self._session_flag(session_name)
            rating = self._rating(best_score, threshold)
            prev = self._prev_candle

            # SMC booleans
            has_bos = dom_smc["structure"]["bos"]
            has_choch = dom_smc["structure"]["choch"]
            has_fvg = dom_smc["fvg"]["in_zone"]
            has_ob = dom_smc["order_block"]["at_zone"]
            has_liq = dom_smc["liquidity"]["swept"]
            scores = self._smc_scores(confluence[dom_key])

            active = []
            if has_choch:
                active.append("CHoCH")
            if has_bos:
                active.append("BOS")
            if has_ob:
                active.append("OB")
            if has_fvg:
                active.append("FVG")
            if has_liq:
                active.append("LiqSweep")

            # Price delta
            if prev:
                pd = price - prev["price"]
                p_em = "\U0001f7e2" if pd >= 0 else "\U0001f534"
                price_str = f"{price:.2f} ({p_em}{pd:+.2f})"
            else:
                price_str = f"{price:.2f}"

            # RSI
            rsi_str = f"{rsi:.0f}"
            if prev:
                r_arr = ""
                if rsi > prev["rsi"]:
                    r_arr = "\u2b06\ufe0f"
                elif rsi < prev["rsi"]:
                    r_arr = "\u2b07\ufe0f"
                rsi_str = f"{prev['rsi']:.0f}\u2192{rsi:.0f}{r_arr}"
            if rsi > 70:
                rsi_str += " OB"
            elif rsi < 30:
                rsi_str += " OS"

            # EMA context
            ema_ctx = "mixed"
            if ema_20 > 0 and ema_50 > 0:
                if price > ema_20 > ema_50:
                    ema_ctx = "bullish"
                elif price < ema_20 < ema_50:
                    ema_ctx = "bearish"
                elif price > ema_20 and ema_20 < ema_50:
                    ema_ctx = "early rev"
                elif price < ema_20 and ema_20 > ema_50:
                    ema_ctx = "pullback"
                else:
                    ema_ctx = "converging"

            # MACD
            macd_str = "\u2014"
            if isinstance(macd_h, (int, float)):
                m_dir = "bull" if macd_h > 0 else ("bear" if macd_h < 0 else "flat")
                m_arr = ""
                if prev and prev.get("macd_h") is not None:
                    if macd_h > prev["macd_h"]:
                        m_arr = "\u2b06\ufe0f"
                    elif macd_h < prev["macd_h"]:
                        m_arr = "\u2b07\ufe0f"
                macd_str = f"{macd_h:+.2f} {m_dir}{m_arr}"

            # H1 bias
            h1_aligned = (
                (h1_bias == "bullish" and best_dir == "LONG")
                or (h1_bias == "bearish" and best_dir == "SHORT")
            )
            h1_tag = "aligned" if h1_aligned else ("counter" if h1_bias != "neutral" else "")

            # Structure marks
            bos_m = "\u2713" if has_bos else "\u2717"
            choch_m = "\u2713" if has_choch else "\u2717"

            # ═══════ BUILD MESSAGE ═══════
            L = []

            # Header
            L.append(f"\U0001f985 <b>XAUUSD</b> \u2014 {session_name.upper()} {flag}")
            L.append(
                f"Keyakinan: <b>{confidence:.1f}/10</b> | "
                f"Outlook: <b>{outlook_label}</b> {outlook_emoji}"
            )

            # [ KESIMPULAN ]
            L.append("")
            L.append("[ KESIMPULAN ]")
            L.append(f"\u203a Regime: {regime.upper()}")
            L.append(f"\u203a Struktur: BOS {bos_m} | CHoCH {choch_m}")

            if has_signal:
                mtf_m = "\u2713" if mtf_aligned else "\u2717"
                h1_info = f"H1: {h1_bias.upper()}"
                if h1_tag:
                    h1_info += f" ({h1_tag})"
                L.append(f"\u203a {h1_info} | MTF: {mtf_m}")
                L.append(
                    f"\u203a Rating: <b>{rating}</b> | "
                    f"Score: <b>{best_score:.2f}</b> / {threshold}"
                )
            else:
                L.append(f"\u203a Score: {best_score:.2f} / {threshold} | Rating: {rating}")

            # [ SINYAL SMC ] — only when signal detected
            if has_signal and active:
                L.append("")
                L.append("[ SINYAL SMC ]")
                # CHoCH + BOS with individual scores
                choch_s = f"\u2713 ({scores['choch']:.2f})" if has_choch else "\u2717"
                bos_s = f"\u2713 ({scores['bos']:.2f})" if has_bos else "\u2717"
                L.append(f"\u203a CHoCH: {choch_s} | BOS: {bos_s}")
                # OB + FVG + Liq
                ob_s = f"\u2713 ({scores['ob']:.2f})" if has_ob else "\u2717"
                fvg_s = f"\u2713 ({scores['fvg']:.2f})" if has_fvg else "\u2717"
                liq_s = f"\u2713 ({scores['liq']:.2f})" if has_liq else "\u2717"
                L.append(f"\u203a OB: {ob_s} | FVG: {fvg_s} | Liq: {liq_s}")

            # [ MARKET ]
            L.append("")
            L.append("[ MARKET ]")
            L.append(f"\U0001f4b9 {price_str} | RSI {rsi_str} | ATR {atr:.2f}")
            L.append(f"\u2514 EMA: {ema_ctx} | MACD: {macd_str} | Vol: {vol_str}")

            # [ VERDICT ]
            L.append("")
            L.append("[ VERDICT ]")
            # Extract tier label from gate_result (populated when signal fires)
            tier_label = (gate_result or {}).get("quality_tier", "")
            tier_badge = ""
            if tier_label:
                if "HIGH" in tier_label or "A:" in tier_label:
                    tier_badge = " \u2b50[A:HIGH]"
                elif "MED" in tier_label or "B:" in tier_label:
                    tier_badge = " [B:MED]"
                elif "LOW" in tier_label or "C:" in tier_label:
                    tier_badge = " [C:LOW]"

            if has_signal:
                smc_combo = " + ".join(active) if active else "Tech"
                tags = []
                tl = trend_str.lower()
                if (best_dir == "LONG" and "bull" in tl) or (best_dir == "SHORT" and "bear" in tl):
                    tags.append("trend \u2713")
                elif "neutral" not in tl and "rang" not in tl:
                    tags.append("CT")
                if mtf_aligned:
                    tags.append("MTF \u2713")
                tag_s = " | ".join(tags)
                L.append(f"\u2705 <b>{best_dir}</b> \u2014 {smc_combo}{tier_badge}")
                if tag_s:
                    L.append(f"\u2192 {tag_s} \u2192 CHECKING ENTRY GATES...")
                else:
                    L.append(f"\u2192 CHECKING ENTRY GATES...")
            else:
                gap = threshold - best_score
                missing = []
                if not has_choch and not has_bos:
                    missing.append("CHoCH/BOS")
                if not has_ob and not has_fvg:
                    missing.append("FVG/OB")
                if not active:
                    missing = ["SMC signal"]
                if gap <= 0.005:
                    # Score is at or above threshold — blocked by floating point
                    # or another gate; don't imply missing SMC signals are the cause.
                    miss_str = f"Score batas ({best_score:.2f}/{threshold:.2f}) — cek gate"
                elif missing:
                    miss_str = f"Butuh {' + '.join(missing[:2])} (+{gap:.2f})"
                else:
                    miss_str = f"Gap: +{gap:.2f}"
                delta_str = ""
                if prev:
                    prev_best = max(prev.get("bull_score", 0), prev.get("bear_score", 0))
                    diff = best_score - prev_best
                    if abs(diff) >= 0.01:
                        arrow = "\u2191" if diff > 0 else "\u2193"
                        delta_str = f" | {arrow}{abs(diff):.2f} vs prev"
                L.append(f"\u23f8 <b>NO TRADE</b> \u2014 {miss_str}{delta_str}")

            # [ POSISI ] — always shown
            L.append("")
            if position_info:
                dir_em = "\U0001f7e2" if position_info["direction"] == "BUY" else "\U0001f534"
                dir_label = "LONG" if position_info["direction"] == "BUY" else "SHORT"
                ticket = position_info["ticket"]
                ep = position_info["entry_price"]
                cp = position_info["current_price"]
                pnl = position_info["profit"]
                rr = position_info.get("rr_current", 0)
                stage = position_info.get("stage", "OPEN")
                sl_val = position_info.get("sl", 0)
                tp_val = position_info.get("tp", 0)
                L.append(f"[ POSISI ] {dir_em} {dir_label} #{ticket}")
                L.append(f"\u203a Entry: {ep:.2f} \u2192 {cp:.2f} ({pnl:+.2f} USD | {rr:+.2f}R)")
                L.append(f"\u203a SL: {sl_val:.2f} | TP: {tp_val:.2f} | Stage: {stage}")
            else:
                L.append("[ POSISI ]")
                L.append("\u203a Tidak ada posisi terbuka")

            # [ ENTRY GATES ] — only when signal detected but gates blocked
            if gate_result is not None and has_signal:
                passed = gate_result.get("passed", False)
                if passed:
                    L.append("")
                    tier_note = f" {tier_badge.strip()}" if tier_badge else ""
                    L.append(f"[ ENTRY GATES ] \u2705{tier_note}")
                    reason = gate_result.get("reason", "")
                    if reason:
                        L.append(f"\u203a {reason}")
                    else:
                        L.append("\u203a All gates passed")
                else:
                    reason = gate_result.get("reason", "")
                    if reason:
                        L.append("")
                        L.append("[ ENTRY GATES ] \u274c")
                        # Parse BULL/BEAR from "BULL: ... | BEAR: ..."
                        bull_raw, bear_raw = "", ""
                        for part in reason.split(" | "):
                            p = part.strip()
                            if p.startswith("BULL:"):
                                bull_raw = p[5:].strip()
                                if "checks failed:" in bull_raw:
                                    bull_raw = bull_raw.split("checks failed:")[-1].strip()
                            elif p.startswith("BEAR:"):
                                bear_raw = p[5:].strip()
                        if bull_raw:
                            items = [self._shorten_gate_reason(r) for r in bull_raw.split(", ")]
                            L.append(f"\U0001f7e2 BUY \u2014 {' | '.join(items)}")
                        if bear_raw:
                            items = [self._shorten_gate_reason(r) for r in bear_raw.split(", ")]
                            L.append(f"\U0001f534 SELL \u2014 {' | '.join(items)}")
                        # Fallback: if no BULL/BEAR parsed, show raw reason
                        if not bull_raw and not bear_raw:
                            L.append(f"\u203a {self._shorten_gate_reason(reason)}")
            elif gate_result is not None and not has_signal:
                # No signal detected — gate_result may have a warmup/info reason
                reason = gate_result.get("reason", "")
                if reason and "warmup" in reason.lower():
                    L.append("")
                    L.append(f"[ INFO ] \u203a {reason}")

            # Footer
            L.append("")
            L.append(f"\u23f0 {self._ts()}")

            msg = "\n".join(L)

            # Store for next comparison
            self._prev_candle = {
                "price": price,
                "atr": atr,
                "rsi": rsi,
                "macd_h": macd_h if isinstance(macd_h, (int, float)) else None,
                "bull_score": bull_score,
                "bear_score": bear_score,
            }

            return self._send(msg)

        except Exception as e:
            self.logger.warning(f"Telegram analysis error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _shorten_gate_reason(self, raw: str) -> str:
        """Shorten verbose gate reasons into compact labels."""
        import re
        r = raw.strip()
        if "structure support" in r:
            return "\u2717 No BOS/CHoCH"
        if "SMC signal" in r:
            # "Only 0 SMC signal(s), need 1+ (position #1)" → "✗ SMC 0/1"
            m = re.search(r"(\d+) SMC.*need (\d+)", r)
            if m:
                return f"\u2717 SMC {m.group(1)}/{m.group(2)}"
            return "\u2717 SMC insufficient"
        if "RSI extreme overbought" in r:
            m = re.search(r"([\d.]+)", r)
            rsi_val = m.group(1) if m else "?"
            return f"\u2717 RSI overbought ({rsi_val})"
        if "RSI extreme oversold" in r:
            m = re.search(r"([\d.]+)", r)
            rsi_val = m.group(1) if m else "?"
            return f"\u2717 RSI oversold ({rsi_val})"
        if "RSI bouncing" in r:
            return "\u2717 RSI bounce block"
        if "Market conditions" in r:
            return "\u2717 Market unfavorable"
        if "MTF not aligned" in r:
            return "\u2717 MTF misaligned"
        if "FVG or Order Block" in r:
            return "\u2717 No FVG/OB"
        if "confluence too low" in r:
            return "\u2717 Confluence too low"
        if "SL cooldown" in r:
            # "SL cooldown BUY (2c)" → "✗ SL cooldown BUY (2c)"
            m = re.search(r"SL cooldown (\w+) \((\d+)c\)", r)
            if m:
                return f"\u23f8 SL cooldown {m.group(1)} ({m.group(2)}c)"
            return "\u23f8 SL cooldown"
        if "Direction limit" in r:
            # "Direction limit (BUY)" → "✗ Dir limit BUY"
            m = re.search(r"Direction limit \((\w+)\)", r)
            if m:
                return f"\u2717 Dir limit {m.group(1)}"
            return "\u2717 Dir limit"
        if "Spacing too close" in r or "Too close to" in r:
            # "Spacing too close #12345 (3.2pts)" → "✗ Spacing 3.2pts"
            m = re.search(r"\(([\d.]+)pts\)", r)
            if m:
                return f"\u2717 Spacing ({m.group(1)}pts)"
            return "\u2717 Spacing too close"
        if "Spacing block" in r:
            return "\u2717 Spacing block"
        if "Micro acct" in r or "micro account" in r.lower():
            return "\u2717 Micro acct limit"
        if "Max position" in r:
            return "\u2717 Max positions"
        if "Order failed" in r:
            # "Order failed (TRADE_RETCODE_ERROR)" → "✗ Order failed"
            m = re.search(r"Order failed \((.{0,20})\)", r)
            if m:
                return f"\u274c Order failed: {m.group(1)}"
            return "\u274c Order failed"
        if "No conditions met" in r:
            return "\u2717 No signal"
        return f"\u2717 {r[:40]}"

    def send_gate_rejection(self, reason: str) -> bool:
        """Deprecated: gate info now included in send_signal_analysis(). No-op."""
        return False

    # ─── Entry Notification ──────────────────────────────────────

    def send_entry(self, direction: str, price: float, sl: float, tp: float,
                   lot: float, ticket: int, confidence: float,
                   smc_signals: str = "", session: str = "",
                   regime: str = "", quality_tier: str = "") -> bool:
        try:
            dir_label = "LONG" if direction == "BUY" else "SHORT"
            dir_em = "\U0001f7e2" if dir_label == "LONG" else "\U0001f534"
            sl_dist = abs(price - sl)
            tp_dist = abs(tp - price)
            rr = tp_dist / sl_dist if sl_dist > 0 else 0

            # Quality tier badge: [A:HIGH] = gold star, [B:MED] = normal, [C:LOW] = skip
            tier_badge = ""
            if quality_tier:
                if "HIGH" in quality_tier or "A:" in quality_tier:
                    tier_badge = " \u2b50 A-GRADE"
                elif "MED" in quality_tier or "B:" in quality_tier:
                    tier_badge = " B-GRADE"

            L = [
                f"{dir_em} <b>ENTRY \u2014 {dir_label}</b> @{price:.2f}{tier_badge}",
                "",
                "[ PARAMETER MISI ] \U0001f512",
                f"\U0001f3af Entry: {price:.2f}",
                f"\U0001f6d1 SL: {sl:.2f} (-{sl_dist:.1f} pts)",
                f"\U0001f48e TP: {tp:.2f} (+{tp_dist:.1f} pts)",
                f"\U0001f4ca R:R 1:{rr:.1f} | Lot: {lot}",
            ]

            L.append("")
            L.append("[ DETAIL ]")
            if smc_signals:
                L.append(f"\u203a Signals: {smc_signals}")
            detail_parts = [f"Score: {confidence * 10:.1f}/10"]
            if quality_tier:
                detail_parts.append(f"Tier: {quality_tier}")
            if session:
                detail_parts.append(f"Session: {session}")
            if regime:
                detail_parts.append(f"Regime: {regime.upper()}")
            L.append("\u203a " + " | ".join(detail_parts))
            L.append("")
            L.append(f"\u23f0 {self._ts()} | #{ticket}")

            return self._send("\n".join(L))

        except Exception as e:
            self.logger.warning(f"Telegram entry error: {e}")
            return False

    # ─── Exit Notification ───────────────────────────────────────

    def send_exit(self, direction: str, ticket: int, entry_price: float,
                  exit_price: float, profit: float, reason: str = "",
                  lot: float = 0.01, sl: float = 0.0,
                  entry_time: str = "") -> bool:
        try:
            pts = abs(exit_price - entry_price) if exit_price and entry_price else 0
            emoji = self._pnl_emoji(profit)
            result = self._pnl_label(profit)
            dir_label = "LONG" if "BUY" in str(direction).upper() else "SHORT"

            # Compute RR if SL is available
            rr_str = ""
            if sl and entry_price:
                sl_dist = abs(entry_price - sl)
                if sl_dist > 0:
                    rr_val = profit / (sl_dist * lot * 100)  # XAUUSD: 100 oz/lot
                    rr_str = f" | RR: {rr_val:+.2f}R"

            # Duration if entry_time is available
            dur_str = ""
            if entry_time:
                try:
                    from datetime import datetime, timezone
                    entry_dt = datetime.strptime(entry_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    dur_min = int((datetime.now(timezone.utc) - entry_dt).total_seconds() / 60)
                    if dur_min < 60:
                        dur_str = f" | {dur_min}min"
                    else:
                        dur_str = f" | {dur_min//60}h{dur_min%60:02d}m"
                except Exception:
                    pass

            L = [
                f"{emoji} <b>{result}</b> {profit:+.2f} USD | {dir_label} #{ticket}",
                f"\u203a {entry_price:.2f} \u2192 {exit_price:.2f} ({pts:.1f} pts{rr_str}){dur_str}",
                f"\u203a {reason}",
                "",
                f"\u23f0 {self._ts()}",
            ]

            return self._send("\n".join(L))

        except Exception as e:
            self.logger.warning(f"Telegram exit error: {e}")
            return False

    # ─── Position Modification ───────────────────────────────────

    def send_modification(self, ticket: int, action: str, details: str = "") -> bool:
        try:
            msg = (
                f"\u2699\ufe0f <b>{action}</b> | #{ticket}\n"
                f"\u203a {details}\n"
                f"\u203a {self._ts_short()}"
            )
            return self._send(msg)
        except Exception as e:
            self.logger.warning(f"Telegram modify error: {e}")
            return False

    # ─── Typed Exit Notifications ────────────────────────────────

    def send_session_exit(self, ticket: int, profit: float, reason: str,
                          minutes_to_close: int = 0) -> bool:
        try:
            emoji = self._pnl_emoji(profit)
            L = [
                f"\U0001f553 <b>SESSION EXIT</b> {emoji} {profit:+.2f} USD | #{ticket}",
                f"\u203a {reason} | NY Close in {minutes_to_close}min",
                "",
                f"\u23f0 {self._ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            self.logger.warning(f"Telegram session exit error: {e}")
            return False

    def send_recovery_exit(self, ticket: int, profit: float,
                           reason: str = "") -> bool:
        try:
            emoji = self._pnl_emoji(profit)
            L = [
                f"\U0001f6e1 <b>RECOVERY EXIT</b> {emoji} {profit:+.2f} USD | #{ticket}",
                f"\u203a {reason}",
                "",
                f"\u23f0 {self._ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            self.logger.warning(f"Telegram recovery exit error: {e}")
            return False

    def send_early_profit_exit(self, ticket: int, profit: float,
                               reason: str) -> bool:
        try:
            L = [
                f"\U0001f4b0 <b>EARLY PROFIT</b> {profit:+.2f} USD | #{ticket}",
                f"\u203a {reason}",
                "",
                f"\u23f0 {self._ts()}",
            ]
            return self._send("\n".join(L))
        except Exception as e:
            self.logger.warning(f"Telegram early profit error: {e}")
            return False

    def send_recovery_action(self, ticket: int, action: str,
                             details: str = "") -> bool:
        try:
            msg = (
                f"\U0001f504 <b>RECOVERY: {action.upper()}</b> | #{ticket}\n"
                f"\u203a {details}\n"
                f"\u203a {self._ts_short()}"
            )
            return self._send(msg)
        except Exception as e:
            self.logger.warning(f"Telegram recovery action error: {e}")
            return False

    def send_trade_stats(self, analysis: dict) -> bool:
        """
        Send trade history summary in clean section format.
        Accepts dict from trade_analyzer.get_full_analysis().
        """
        try:
            overall = analysis.get("overall", {})
            total = overall.get("total_trades", 0)

            L = ["\U0001f4ca <b>XAUUSD \u2014 TRADE SUMMARY</b>"]

            if total == 0:
                L.append("\u203a Belum ada riwayat trade tertutup.")
                L.append("")
                L.append(f"\u23f0 {self._ts()}")
                return self._send("\n".join(L))

            wins = overall.get("wins", 0)
            losses = overall.get("losses", 0)
            bes = overall.get("breakevens", 0)
            wr = overall.get("win_rate", 0)
            pf = overall.get("profit_factor", 0)
            net = overall.get("net_profit", 0)
            avg_win = overall.get("avg_win", 0)
            avg_loss = abs(overall.get("avg_loss", 0))

            pnl_em = "\U0001f7e2" if net >= 0 else "\U0001f534"

            avg_dur = overall.get("avg_duration_minutes")
            max_cl = overall.get("max_consecutive_losses", 0)
            best_t = overall.get("best_trade")
            worst_t = overall.get("worst_trade")

            # [ PERFORMA ]
            L.append("")
            L.append("[ PERFORMA ]")
            L.append(f"\u203a {total} trades | W: {wins} / L: {losses} / BE: {bes}")
            L.append(f"\u203a Win Rate: <b>{wr:.0%}</b> | PF: <b>{pf:.2f}</b>")
            L.append(f"\u203a Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
            L.append(f"\u203a Net P/L: {pnl_em} <b>${net:+.2f}</b>")
            if avg_dur is not None:
                L.append(f"\u203a Avg Hold: {avg_dur:.0f} min")
            if max_cl > 0:
                L.append(f"\u203a Max Consec Loss: {max_cl}x \u26a0\ufe0f")
            if best_t:
                L.append(f"\u203a Best: +${best_t['profit']:.2f} ({best_t['combo']})")
            if worst_t:
                L.append(f"\u203a Worst: ${worst_t['profit']:.2f} ({worst_t['combo']})")

            # [ PER SESI ]
            by_session = analysis.get("by_session", {})
            if by_session:
                L.append("")
                L.append("[ PER SESI ]")
                for sname, s in by_session.items():
                    s_net = s.get("net_profit", 0)
                    s_em = "\U0001f7e2" if s_net >= 0 else "\U0001f534"
                    pf_val = s['profit_factor']
                    pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"
                    L.append(
                        f"\u203a {sname}: {s['wins']}/{s['total']} "
                        f"({s['win_rate']:.0%}) | PF: {pf_str} | "
                        f"{s_em} ${s_net:+.2f}"
                    )

            # [ BY REGIME ]
            by_regime = analysis.get("by_regime", {})
            regime_known = {k: v for k, v in by_regime.items() if k != "Unknown"}
            if regime_known:
                L.append("")
                L.append("[ BY REGIME ]")
                for reg, rs in list(regime_known.items())[:4]:
                    r_net = rs.get("net_profit", 0)
                    r_em = "\U0001f7e2" if r_net >= 0 else "\U0001f534"
                    L.append(
                        f"\u203a {reg}: {rs['wins']}/{rs['total']} "
                        f"({rs['win_rate']:.0%}) | {r_em} ${r_net:+.2f}"
                    )

            # [ TOP COMBO SMC ]
            by_combo = analysis.get("by_smc_combo", {})
            if by_combo:
                L.append("")
                L.append("[ TOP COMBO SMC ]")
                for combo, cs in list(by_combo.items())[:5]:
                    c_net = cs.get("net_profit", 0)
                    c_em = "\U0001f7e2" if c_net >= 0 else "\U0001f534"
                    L.append(
                        f"\u203a {combo}: {cs['wins']}/{cs['total']} "
                        f"({cs['win_rate']:.0%}) | {c_em} ${c_net:+.2f}"
                    )

            L.append("")
            L.append(f"\u23f0 {self._ts()}")
            return self._send("\n".join(L))

        except Exception as e:
            self.logger.warning(f"Telegram trade stats error: {e}")
            return False

    # ─── Bot Status ──────────────────────────────────────────────

    def send_bot_status(self, status: str, details: str = "") -> bool:
        """Generic bot status (used for BOT STOPPED, ORDER FAILED, etc.)."""
        try:
            L = [f"\U0001f916 <b>{status}</b>"]
            if details:
                L.append(details)
            L.append("")
            L.append(f"\u23f0 {self._ts()}")
            return self._send("\n".join(L))
        except Exception as e:
            self.logger.warning(f"Telegram status error: {e}")
            return False

    def send_bot_started(
        self,
        balance: float,
        equity: float,
        scorer_on: bool,
        be_rr: float,
        pc_rr: float,
        max_pos: int,
        is_smc_v4: bool,
        session_check: dict,
        trade_analysis: Optional[dict] = None,
        sl_atr_range: Optional[tuple] = None,
        tp_atr: Optional[float] = None,
    ) -> bool:
        """
        Send clean BOT STARTED message with config, account, market status,
        and a brief last-30-day trade summary.

        Format mirrors the signal analysis aesthetic (section blocks).
        """
        try:
            is_allowed = session_check.get("allowed", False)
            mkt_status = session_check.get("status", "OPEN" if is_allowed else "CLOSED")
            mkt_reason = session_check.get("reason", "")
            opens_in   = session_check.get("opens_in_minutes")

            # Market status line
            if is_allowed:
                mkt_em   = "\U0001f7e2"   # green
                mkt_line = f"{mkt_em} <b>OPEN</b>"
                if mkt_reason:
                    mkt_line += f" \u2014 {mkt_reason}"
            elif mkt_status == "MAINTENANCE":
                mkt_em   = "\U0001f527"   # wrench
                mkt_line = f"{mkt_em} <b>MAINTENANCE</b>"
                if opens_in is not None:
                    h, m = divmod(int(opens_in), 60)
                    eta = f"{h}j {m:02d}m" if h > 0 else f"{m}m"
                    mkt_line += f" \u2022 Buka dalam {eta}"
            else:
                mkt_em   = "\U0001f534"   # red
                mkt_line = f"{mkt_em} <b>CLOSED</b>"
                if mkt_reason:
                    mkt_line += f" \u2014 {mkt_reason}"
                # Note: mkt_reason from session_detector already includes ETA
                # e.g. "Weekend — market opens Sunday 23:00 UTC (in 23h 14m)"
                # so we do NOT append a duplicate Indonesian ETA here.

            smc_label = "V4 Library" if is_smc_v4 else "V3 Custom"
            scorer_label = "ON (V3 Adaptive)" if scorer_on else "OFF (V2 Fixed)"

            L = []
            L.append(f"\U0001f916 <b>XAUUSD \u2014 BOT STARTED</b>")
            L.append(f"Scorer: {scorer_label} | SMC: {smc_label}")

            # [ KONFIGURASI ]
            L.append("")
            L.append("[ KONFIGURASI ]")
            L.append(f"\u203a BE: {be_rr:.2f}R \u2192 Partial: {pc_rr:.2f}R \u2192 Trail")
            if sl_atr_range and tp_atr:
                L.append(
                    f"\u203a SL: {sl_atr_range[0]:.1f}x\u2013{sl_atr_range[1]:.1f}x ATR"
                    f" | TP: {tp_atr:.1f}x ATR (regime-adaptive)"
                )
            elif sl_atr_range:
                L.append(f"\u203a SL: {sl_atr_range[0]:.1f}x\u2013{sl_atr_range[1]:.1f}x ATR (regime-adaptive)")
            L.append(f"\u203a Max Posisi: {max_pos} | TF: M15 | Symbol: XAUUSDm")

            # [ AKUN ]
            L.append("")
            L.append("[ AKUN ]")
            eq_delta = equity - balance
            eq_em = "\U0001f7e2" if eq_delta >= 0 else "\U0001f534"
            L.append(
                f"\U0001f4b0 Balance: <b>${balance:,.2f}</b> | "
                f"Equity: {eq_em} ${equity:,.2f} ({eq_delta:+.2f})"
            )
            L.append(f"\U0001f4ca Market: {mkt_line}")

            # [ LAST 30 HARI ] — from trade_analysis
            if trade_analysis:
                overall = trade_analysis.get("overall", {})
                total = overall.get("total_trades", 0)
                if total > 0:
                    _wins  = overall.get("wins", 0)
                    _loss  = overall.get("losses", 0)
                    _bes   = overall.get("breakevens", 0)
                    wr     = overall.get("win_rate", 0)
                    pf     = overall.get("profit_factor", 0)
                    net    = overall.get("net_profit", 0)
                    net_em = "\U0001f7e2" if net >= 0 else "\U0001f534"
                    pf_str = f"{pf:.2f}" if pf != float("inf") else "\u221e"
                    decisive = _wins + _loss
                    dec_wr = _wins / decisive if decisive > 0 else 0
                    avg_dur = overall.get("avg_duration_minutes")
                    max_cl  = overall.get("max_consecutive_losses", 0)
                    L.append("")
                    L.append("[ LAST 30 HARI ]")
                    L.append(f"\u203a {total} trades \u2014 W: {_wins} / L: {_loss} / BE: {_bes}")
                    L.append(f"\u203a WR: {wr:.0%} overall | {dec_wr:.0%} decisive | PF: {pf_str}")
                    net_line = f"\u203a Net: {net_em} <b>${net:+.2f}</b>"
                    if avg_dur is not None:
                        net_line += f" | Hold: {avg_dur:.0f}min"
                    L.append(net_line)
                    if max_cl >= 3:
                        L.append(f"\u203a Max streak loss: {max_cl}x \u26a0\ufe0f")

            L.append("")
            L.append(f"\u23f0 {self._ts()}")

            return self._send("\n".join(L))

        except Exception as e:
            self.logger.warning(f"Telegram bot_started error: {e}")
            return False

    def send_market_status(
        self,
        status: str,
        reason: str,
        opens_in_minutes: int = None,
    ) -> bool:
        """
        Send market open/closed/maintenance status notification.

        Args:
            status: "OPEN", "MAINTENANCE", or "WEEKEND"/"CLOSED"
            reason: Human-readable reason string
            opens_in_minutes: Minutes until market reopens (None if open)
        """
        try:
            if status == "OPEN":
                emoji = "\U0001f7e2"   # green circle
                title = "MARKET OPEN"
            elif status == "MAINTENANCE":
                emoji = "\U0001f527"   # wrench
                title = "MAINTENANCE BREAK"
            else:
                emoji = "\U0001f534"   # red circle
                title = "MARKET CLOSED"

            L = [f"{emoji} <b>{title}</b>"]
            L.append(f"\u203a {reason}")

            if opens_in_minutes is not None and status != "OPEN":
                h, m = divmod(int(opens_in_minutes), 60)
                if h > 0:
                    L.append(f"\u203a Resumes in: {h}h {m:02d}m")
                else:
                    L.append(f"\u203a Resumes in: {m}m")

            L.append("")
            L.append(f"\u23f0 {self._ts()}")
            return self._send("\n".join(L))
        except Exception as e:
            self.logger.warning(f"Telegram market status error: {e}")
            return False
