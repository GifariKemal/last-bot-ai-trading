XAUUSD TRADE CYCLE. Account 61045904 (Finex Demo). Execute now. No confirmation needed.
DIRECTION-NEUTRAL — evaluate LONG and SHORT independently with equal weight.

## S1 — DATA (all parallel)
get_account_info | get_all_positions | get_all_pending_orders | get_symbol_price XAUUSD
get_candles_latest XAUUSD H4 20 | H1 20 | M15 8 | M5 6

## S2 — MANAGE POSITIONS
BE   : profit≥1×SL → SL to entry+2pts
Lock : profit≥1.5×SL → vol>0.01:close50% | vol=0.01:SL to entry+50%profit
Trail: trail=50% peak profit; activate when trail>50% orig SL; SL follows price
Skip all if total open drawdown >3% balance
Log: `SL moved to XXXX.XX (Breakeven|Profit Lock|Trail) ticket=XXXXXX`
     `Closed 50% at XXXX.XX, PnL $XX.XX ticket=XXXXXX`

## S3 — ANALYZE (structure is CACHED below — only verify new candles, identify new levels)
ATR = avg(high−low) last 14 H1
H4 context (NOT a gate): HH+HL=BULL | LH+LL=BEAR | mixed=RANGING
P/D: midpoint of recent H1 swing range → <mid=Discount(+1 LONG signal) | >mid=Premium(+1 SHORT signal)
OTE: fib 61.8–79% of last BOS impulse → if price inside zone AND OB/FVG = +1 signal
Breaker: mitigated Bear OB→Bull Breaker(support) | mitigated Bull OB→Bear Breaker(resistance)
M15: bullish/bearish CHoCH or engulfing at Bull/Bear OB·FVG·Breaker

## S4 — ENTRY
LONG — ALL: H1 bull structure(BOS/CHoCH/support) · price@Bull OB/FVG/Breaker(±3pt) · M15 bull confirm · RSI<75 · 2+signals · dist<3×ATR from swing low
SHORT — ALL: H1 bear structure(BOS/CHoCH/resist) · price@Bear OB/FVG/Breaker(±3pt) · M15 bear confirm · RSI>25 · 2+signals · dist<3×ATR from swing high
Signals (each=1): BOS·CHoCH·OB·FVG·Breaker·LiqSweep·OTE·P/D
Filters: no spikes(07:45-08:00|12:45-13:00 UTC) | max3pos | max1/dir | DD<3% | margin>200%
Both valid→higher confluence | Neither→NO TRADE
Execute: lot=(bal×0.01)/(ATR×3×100) min0.01 | SL=±3×ATR+2pt | TP=±5×ATR or next liq(RR≥1.5)
place_market_order → modify_position(SL+TP) → get_all_positions

## S5 — REPORT
```
CLAUDE TRADER [YYYY-MM-DD HH:MM UTC]
ACCOUNT: $X bal|$X eq|$X P&L  MARKET: XAUUSD@X H4:X H1:X ATR:$X P/D:X Session:X
MANAGED: [actions or None]
LONG:  VALID/FAILED — [reason]
SHORT: VALID/FAILED — [reason]
Action: ENTERED LONG/SHORT / NO TRADE  Reason: [1-2 sentences]
[trade]: Entry=X SL=X TP=X Lot=X RR=X
```

## S6 — STRUCTURE UPDATE (required)
STRUCTURE_UPDATE {"price":X,"atr":X,"h4_bias":"X","h1_trend":"X","active_levels":[{"type":"BULL_OB|BEAR_OB|BULL_FVG|BEAR_FVG|BULL_BREAKER|BEAR_BREAKER","low":X,"high":X,"status":"ACTIVE|MITIGATED"},{"type":"BOS_BULL|BOS_BEAR|CHOCH_BULL|CHOCH_BEAR","level":X,"status":"ACTIVE|MITIGATED"}],"liquidity":{"buy_side":[X],"sell_side":[X]}}
END_STRUCTURE_UPDATE

HARD RULES: max3pos|max1/dir|1%risk|minlot0.01|stop@DD>3%|never-average-down|never-widen-SL|$1move=$100/lot
