import pandas as pd
from datetime import datetime
import numpy as np


import pandas_market_calendars as mcal

def load_raw_feed(path: str) -> pd.DataFrame:
    """
    - Read a Databento `.csv` 
    - Keep only these columns: `ts_recv` (int64 nanoseconds), `action` (ADD/CANCEL/TRADE), `side` (BID/ASK), `price` (int64 in price ticks), `size` (int64), and `bid_sz_00..03` / `ask_sz_00..03` (top-4 queue snapshots).
    - Discard the first and last 30 minutes of each trading day to exclude open/close artefacts (keep 10:00–15:30 ET).
    - Return a dataframe sorted ascending by `ts_recv`.
    """
    df = pd.read_csv(path)
    df = filter_trading_hours(df, "ts_recv")

NON_PRINTABLE_FLAG = 64  # F_TOB, bit 6 — Trade is non-printable


def orderbook_preprocess(df: pd.DataFrame):
    """
    - Group the event_log by 'ts_recv'
    - Calculate the best ask/bid
    - Find the delta between consecutive book states
    - Aggregate trades, creates

    Returns
    -------
    book_states : dict  {ts -> {key: (price, size)}}
    delta_df    : pd.DataFrame  columns = [ts, level, size, action]
    """
    grouped = df.groupby("ts_recv", sort=False)

    book_states = {}

    # O1: rolling 2-slot buffer — delta computation never touches book_states for lookups
    prev_state = None

    delta = []

    for ts, temp_df in grouped:
        last = temp_df.iloc[-1]

        # Record valid end-of-event snapshots that are not trades
        if (last["flags"] in (128, 130)) and (last["action"] != "T"):
            # O4: tuples instead of lists — immutable, slightly cheaper to allocate
            current_state = {
                "bid_03": (last["bid_px_03"], last["bid_sz_03"]),
                "bid_02": (last["bid_px_02"], last["bid_sz_02"]),
                "bid_01": (last["bid_px_01"], last["bid_sz_01"]),
                "bid_00": (last["bid_px_00"], last["bid_sz_00"]),
                "ask_00": (last["ask_px_00"], last["ask_sz_00"]),
                "ask_01": (last["ask_px_01"], last["ask_sz_01"]),
                "ask_02": (last["ask_px_02"], last["ask_sz_02"]),
                "ask_03": (last["ask_px_03"], last["ask_sz_03"]),
            }
            book_states[ts] = current_state
        else:
            continue

        # --- Delta computation ---
        n_side_mask = temp_df["side"] == "N"
        if n_side_mask.any():
            # B3: use the actual N-side row, not blindly iloc[-1]
            n_row = temp_df[n_side_mask].iloc[-1]

            if n_row.flags == 0:
                prev_state = current_state
                continue

            if n_row["action"] != "F":
                if prev_state is None:
                    prev_state = current_state
                    continue

                prev_prices  = [v[0] for v in prev_state.values()]
                prev_sizes   = [v[1] for v in prev_state.values()]
                curr_prices  = [v[0] for v in current_state.values()]
                curr_sizes   = [v[1] for v in current_state.values()]

                if prev_prices == curr_prices:
                    # Only sizes changed — no price level shift
                    for i, (ps, cs) in enumerate(zip(prev_sizes, curr_sizes)):
                        change = cs - ps
                        if change != 0:
                            delta.append([ts, queue_level(i), change, "C" if change < 0 else "A"])
                else:
                    prev_ask    = prev_prices[4:]
                    curr_ask    = curr_prices[4:]
                    prev_ask_sz = prev_sizes[4:]
                    curr_ask_sz = curr_sizes[4:]

                    # Reverse bids so index 0 = best bid
                    prev_bid    = prev_prices[:4][::-1]
                    curr_bid    = curr_prices[:4][::-1]
                    prev_bid_sz = prev_sizes[:4][::-1]
                    curr_bid_sz = curr_sizes[:4][::-1]

                    delta += scan_side(ts, prev_ask, curr_ask, prev_ask_sz, curr_ask_sz, "ask")
                    delta += scan_side(ts, prev_bid, curr_bid, prev_bid_sz, curr_bid_sz, "bid")

        else:
            # B4: drop the unreliable flag-count condition — T+C action pattern is sufficient
            is_normal_trade = (
                len(temp_df) >= 2
                and temp_df.iloc[-1]["action"] == "C"
                and temp_df.iloc[-2]["action"] == "T"
            )
            rows_to_process = temp_df.iloc[:-1] if is_normal_trade else temp_df

            for row in rows_to_process.itertuples(index=False):
                # B5: skip non-printable trade records
                if row.action == "T" and (row.flags & NON_PRINTABLE_FLAG):
                    continue
                level = (row.depth + 1) if row.side == "A" else -(row.depth + 1)
                delta.append([ts, level, row.size, row.action])

        # O1: roll the buffer forward
        prev_state = current_state

    return book_states, pd.DataFrame(delta, columns=["ts", "level", "size", "action"])


def queue_level(i: int) -> int:
    """Map flat index 0-7 → queue level: -4, -3, -2, -1, +1, +2, +3, +4."""
    return i - 4 if i < 4 else i - 3


def scan_side(ts, prev_px, curr_px, prev_sz, curr_sz, side):
    """
    Scan from best price outward.
    First price difference determines the event type for that level and beyond.

    Returns list of [ts, queue_level, size_change, action].
    """
    result = []
    n = len(prev_px)

    # O2: precompute set once — used in both branches below
    prev_set = set(prev_px)

    # O3: level helper — eliminates repeated ternary across all branches
    def _level(i):
        return (i + 1) if side == "ask" else -(i + 1)

    # Find first position where price differs
    first_diff = next((i for i in range(n) if prev_px[i] != curr_px[i]), None)

    if first_diff is None:
        # No price change — only size changes
        for i, (ps, cs) in enumerate(zip(prev_sz, curr_sz)):
            change = cs - ps
            if change != 0:
                result.append([ts, _level(i), change, "C" if change < 0 else "A"])
        return result

    prev_price    = prev_px[first_diff]
    current_price = curr_px[first_diff]
    price_improved = (current_price < prev_price) if side == "ask" else (current_price > prev_price)

    if price_improved:
        if current_price not in prev_set:
            # New best price inserted inside the spread — CREATE event
            result.append([ts, _level(first_diff), curr_sz[first_diff],
                           "CREATE_ASK" if side == "ask" else "CREATE_BID"])
            # All levels from first_diff+1 shifted one step deeper
            for i in range(first_diff + 1, n):
                change = curr_sz[i] - prev_sz[i - 1]
                if change != 0:
                    result.append([ts, _level(i), change, "C" if change < 0 else "A"])
        else:
            # Best level consumed — everything shifted inward
            result.append([ts, _level(first_diff), -prev_sz[first_diff], "C"])
            for i in range(first_diff + 1, n):
                prev_idx = i + 1
                if prev_idx < n:
                    change = curr_sz[i] - prev_sz[prev_idx]
                    if change != 0:
                        result.append([ts, _level(i), change, "C" if change < 0 else "A"])
                else:
                    # Newly revealed deepest level
                    result.append([ts, _level(i), curr_sz[i], "A"])
    else:
        if current_price not in prev_set:
            # New gap level inserted between existing levels
            result.append([ts, _level(first_diff), curr_sz[first_diff], "A"])
            for i in range(first_diff + 1, n):
                prev_idx = i - 1
                if prev_idx < n and prev_px[prev_idx] == curr_px[i]:
                    change = curr_sz[i] - prev_sz[prev_idx]
                    if change != 0:
                        result.append([ts, _level(i), change, "C" if change < 0 else "A"])
        else:
            result.append([ts, _level(first_diff), -prev_sz[first_diff], "C"])
            for i in range(first_diff + 1, n):
                change = curr_sz[i] - prev_sz[i]
                if change != 0:
                    result.append([ts, _level(i), change, "C" if change < 0 else "A"])

    return result


def convert_book_states(book_states: dict, tick_size: float = 1e7) -> pd.DataFrame:
    """
    Convert book_states dict into a structured DataFrame.

    book_states : {ts -> {"bid_00": (price, size), ..., "ask_00": (price, size), ...}}
    tick_size   : one tick in Databento fixed-point units (1e7 = 1 cent)

    Output columns:
        ts, spread, imbalance, best_size, q-4, q-3, q-2, q-1, q+1, q+2, q+3, q+4,
        best_bid_px, best_ask_px
    """
    rows = []

    for ts, state in book_states.items():
        bid_px = [state[f"bid_0{i}"][0] for i in range(4)]
        bid_sz = [state[f"bid_0{i}"][1] for i in range(4)]
        ask_px = [state[f"ask_0{i}"][0] for i in range(4)]
        ask_sz = [state[f"ask_0{i}"][1] for i in range(4)]

        best_bid = bid_px[0]
        best_ask = ask_px[0]

        if best_bid is None or best_ask is None or pd.isna(best_bid) or pd.isna(best_ask):
            continue

        spread_ticks = int(round((best_ask - best_bid) / tick_size))

        q_bid1, q_ask1 = bid_sz[0], ask_sz[0]
        total     = q_bid1 + q_ask1
        imbalance = (q_bid1 - q_ask1) / total if total > 0 else 0.0

        bid_lookup = {px: sz for px, sz in zip(bid_px, bid_sz) if px is not None and not pd.isna(px)}
        ask_lookup = {px: sz for px, sz in zip(ask_px, ask_sz) if px is not None and not pd.isna(px)}

        def bid_queue(k):
            return bid_lookup.get(best_bid - (k - 1) * tick_size, 0)

        def ask_queue(k):
            return ask_lookup.get(best_ask + (k - 1) * tick_size, 0)

        rows.append({
            "ts":          ts,
            "spread":      spread_ticks,
            "imbalance":   round(imbalance, 4),
            "best_size":   round(bid_queue(1) + ask_queue(1), 4),  # B1: was missing from columns
            "q-4":         round(bid_queue(4), 4),
            "q-3":         round(bid_queue(3), 4),
            "q-2":         round(bid_queue(2), 4),
            "q-1":         round(bid_queue(1), 4),
            "q+1":         round(ask_queue(1), 4),
            "q+2":         round(ask_queue(2), 4),
            "q+3":         round(ask_queue(3), 4),
            "q+4":         round(ask_queue(4), 4),
            "best_bid_px": best_bid * 1e-9,
            "best_ask_px": best_ask * 1e-9,
        })

    # B1: best_size added to columns list
    return pd.DataFrame(rows, columns=[
        "ts", "spread", "imbalance", "best_size",
        "q-4", "q-3", "q-2", "q-1",
        "q+1", "q+2", "q+3", "q+4",
        "best_bid_px", "best_ask_px",
    ])

    
def aggregate_trades(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Group consecutive TRADE rows that share the same `ts_recv` value into one row with summed `size`.
    - Rationale: a single aggressive order hitting multiple resting orders generates one message per fill; these must be treated as one event for volume estimation.
    - Implementation: use `groupby` on a "burst ID" that increments whenever `ts_recv` changes or `action != TRADE`. Take the first row's metadata and sum `size`.
    """
    trades = delta_df[delta_df["action"] == "T"].copy()
    
    # Burst ID: increments when ts changes between consecutive trade rows
    trades["burst_id"] = (trades["ts"] != trades["ts"].shift()).cumsum()
    
    aggregated = trades.groupby("burst_id", sort=False).agg(
        ts     = ("ts",     "first"),
        level  = ("level",  "first"),
        size   = ("size",   "sum"),     # key aggregation
        action = ("action", "first"),
    ).reset_index(drop=True)
    
    non_trades = delta_df[delta_df["action"] != "T"]
    return pd.concat([aggregated, non_trades]).sort_values("ts").reset_index(drop=True)


def aggregate_creates(delta_df: list) -> list:
    """
    Aggregate CREATE events with any Add events at the same ts and level.
    When a new price level is created, other participants immediately join it —
    those Add events at the same ts and level should be folded into the Create.
    
    Also aggregates consecutive Add events at the same ts and level that
    follow a Create, even if they appear non-consecutively among other events,
    as long as they share the same ts.
    """
    creates = delta_df[delta_df["action"].isin(["CREATE_ASK", "CREATE_BID"])].copy()
    others  = delta_df[~delta_df["action"].isin(["CREATE_ASK", "CREATE_BID"])].copy()

    # Burst ID: increments when ts, level, or action changes
    creates["burst_id"] = (
        (creates["ts"]     != creates["ts"].shift())     |
        (creates["level"]  != creates["level"].shift())  |
        (creates["action"] != creates["action"].shift())
    ).cumsum()

    aggregated = creates.groupby("burst_id", sort=False).agg(
        ts     = ("ts",     "first"),
        level  = ("level",  "first"),
        size   = ("size",   "sum"),     # accumulate all participants joining new level
        action = ("action", "first"),
    ).reset_index(drop=True)

    return pd.concat([aggregated, others]).sort_values("ts").reset_index(drop=True)

def save_processed(df: pd.DataFrame, ticker: str):
    """
    - Write the cleaned, annotated dataframe to `data/processed/{ticker}_events.parquet`.
    - Schema: `[ts_recv, action, side, level, price, size, vol_norm, dt_ns, log10_dt, imb_bin, spread_bin, is_fast]`.
    """
    


def filter_trading_hours(df: pd.DataFrame, ts_col: str = "ts_recv") -> pd.DataFrame:
    """
    filter the first and last 30 min of each trading day
    remaining: 10:00~15:00 each trading day
    """
    # Convert nanosecond UTC to ET
    ts_et = pd.to_datetime(df[ts_col], unit="ns", utc=True).dt.tz_convert("America/New_York")
    
    # Get the date range covered by the data
    start_date = ts_et.dt.date.min()
    end_date   = ts_et.dt.date.max()
    
    # Get the NYSE trading calendar for that range
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # schedule gives you market_open and market_close in UTC for each trading day
    # but we will use fixed 9:30-15:30 ET since regular session times don't change
    trading_dates = set(schedule.index.date)
    
    # Build mask: must be a trading day AND within session hours
    date_only = ts_et.dt.date
    time_only = ts_et.dt.time
    
    market_open  = pd.Timestamp("10:00:00").time()
    market_close = pd.Timestamp("15:30:00").time()
    
    is_trading_day   = date_only.apply(lambda d: d in trading_dates)
    is_trading_hours = (time_only >= market_open) & (time_only <= market_close)
    
    mask = is_trading_day & is_trading_hours
    
    return df[mask].reset_index(drop=True)