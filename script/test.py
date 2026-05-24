import pandas as pd
import numpy as np
from tqdm import tqdm

F_END = 130
N_LEVELS = 10


# ---------- helpers ----------

def book_tuple(row):
    """Hashable top-N book snapshot."""
    return tuple(
        int(row[f'{side}_{f}_{i:02d}'])
        for side in ('bid', 'ask')
        for f in ('px', 'sz')
        for i in range(N_LEVELS)
    )

def visible_size_at(row, price, side):
    """Return size at `price` on `side` ('A' or 'B') in the row's snapshot, or 0.
       Returns 0 if the price is not present at any level on that side."""
    prefix = 'ask' if side == 'A' else 'bid'
    for i in range(N_LEVELS):
        if int(row[f'{prefix}_px_{i:02d}']) == int(price):
            return int(row[f'{prefix}_sz_{i:02d}'])
    return 0

def resting_side(trade_side):
    """For a T row, flip aggressor side to the side where size is removed.
       Databento convention: T.side = aggressor; resting orders are on the opposite side."""
    if trade_side == 'A':
        return 'B'
    if trade_side == 'B':
        return 'A'
    return None  # 'N' or anything else

def find_sessions(df):
    sessions = []
    i, n = 0, len(df)
    pbar = tqdm(total=n, desc="Scanning sessions", leave=False)
    while i < n:
        if df.at[i, 'action'] == 'T':
            ts = df.at[i, 'ts_recv']
            j = i
            while j < n and df.at[j, 'action'] == 'T' and df.at[j, 'ts_recv'] == ts:
                j += 1
            # Session = pure T-run; don't try to grab a trailing C.
            sessions.append((i, j - 1, 'trade'))
            pbar.update(j - i)
            i = j
        else:
            sessions.append((i, i, 'single'))
            pbar.update(1)
            i += 1
    pbar.close()
    return sessions

def has_pending_trade(df, idx):
    """True iff idx is immediately preceded by T rows not terminated by C/F_END."""
    j = idx - 1
    saw_trade = False
    while j >= 0 and df.at[j, 'action'] == 'T':
        saw_trade = True
        j -= 1
    if saw_trade and j >= 0:
        if df.at[j, 'action'] == 'C' and df.at[j, 'flags'] == F_END:
            return False
    return saw_trade

# ---------- claim validators ----------

def validate_claim_1(df, sessions):
    """
    Empirical version: for each pure-T session, find which visible level
    at the trade's price actually changed by sum(T sizes). If exactly one
    side matches, claim passes.
    """
    results, viol_close = [], []
    for start, end, kind in tqdm(sessions, desc="Claim 1", leave=False):
        if kind != 'trade':
            continue
        block = df.iloc[start:end + 1]
        if (block['side'] == 'N').any():
            continue
        if start == 0 or end + 1 >= len(df):
            continue

        t_rows = block[block['action'] == 'T']
        price = int(t_rows.iloc[0]['price'])
        total_t_size = int(t_rows['size'].sum())

        pre_row  = df.iloc[start - 1]
        post_row = df.iloc[end + 1]

        d_bid = visible_size_at(pre_row, price, 'B') - visible_size_at(post_row, price, 'B')
        d_ask = visible_size_at(pre_row, price, 'A') - visible_size_at(post_row, price, 'A')

        bid_match = (d_bid == total_t_size)
        ask_match = (d_ask == total_t_size)
        either_match = bid_match or ask_match

        results.append({
            'start': start, 'end': end, 'price': price,
            'sum_T_size': total_t_size,
            'd_bid': d_bid, 'd_ask': d_ask,
            'match': either_match,
        })

    summary = pd.DataFrame(results)
    print("=== Claim 1 (empirical): non-N trade sessions ===")
    print(f"  sessions examined: {len(summary)}")
    if len(summary):
        print(f"  size matches (bid OR ask): {summary['match'].sum()}/{len(summary)} "
              f"({summary['match'].mean()*100:.1f}%)")
        miss = summary[~summary['match']]
        if len(miss):
            print("  first mismatches:")
            print(miss.head())
    return summary

def validate_claim_2(df, sessions):
    """
    Claim 2 (revised): T rows with side='N' don't affect the visible book.
    Test: visible size at the trade's price on BOTH sides should be unchanged
    from the row before the trade session to the row after it.
    """
    rows = []
    for start, end, kind in tqdm(sessions, desc="Claim 2", leave=False):
        if kind != 'trade':
            continue
        block = df.iloc[start:end + 1]
        n_trades = block[(block['action'] == 'T') & (block['side'] == 'N')]
        if len(n_trades) == 0:
            continue

        # Need pre and post rows. Pre = row before session, post = closer (end row).
        if start == 0:
            continue
        pre_row = df.iloc[start - 1]
        post_row = df.iloc[end+1]

        # For each N-side trade in the session, check both sides at its price.
        for _, r in n_trades.iterrows():
            price = int(r['price'])
            pre_bid = visible_size_at(pre_row, price, 'B')
            pre_ask = visible_size_at(pre_row, price, 'A')
            post_bid = visible_size_at(post_row, price, 'B')
            post_ask = visible_size_at(post_row, price, 'A')

            book_touched = (pre_bid != post_bid) or (pre_ask != post_ask)
            rows.append({
                'idx': r.name,
                'price': price,
                'pre_bid_sz': pre_bid, 'post_bid_sz': post_bid,
                'pre_ask_sz': pre_ask, 'post_ask_sz': post_ask,
                'book_touched': book_touched,
            })

    res = pd.DataFrame(rows)
    print("\n=== Claim 2 (revised): T side='N' shouldn't touch visible book ===")
    print(f"  N-side trade rows:                 {len(res)}")
    if len(res):
        untouched = (~res['book_touched']).sum()
        print(f"  visible book unchanged (claim 2 holds): {untouched}/{len(res)} "
              f"({untouched / len(res) * 100:.1f}%)")
        print(f"  visible book changed (violates):        {res['book_touched'].sum()}")
        if res['book_touched'].any():
            print("  first real violations:")
            print(res[res['book_touched']].head())
    return res

def validate_claim_3a(df, sessions):
    """
    Claim 3a: A non-trade side='N' row preceded by an unterminated T-run means
    the closing C is missing AND the size of both the trades and this row's
    event got folded into the book at this row. Book delta at the trade's
    price on the resting side should be >= sum(T sizes).
    """
    results = []
    for start, end, kind in tqdm(sessions, desc="Claim 3a", leave=False):
        if kind != 'single':
            continue
        i = start
        row = df.iloc[i]
        if row['side'] != 'N' or row['action'] == 'T':
            continue
        if not has_pending_trade(df, i):
            continue

        j = i - 1
        while j >= 0 and df.at[j, 'action'] == 'T':
            j -= 1
        trade_start = j + 1
        t_rows = df.iloc[trade_start:i]
        if len(t_rows) == 0:
            continue
        agg_side = t_rows.iloc[0]['side']
        book_side = resting_side(agg_side)
        price = int(t_rows.iloc[0]['price'])
        total_t_size = int(t_rows['size'].sum())

        if trade_start == 0 or book_side is None:
            continue
        pre_sz = visible_size_at(df.iloc[trade_start - 1], price, book_side)
        post_sz = visible_size_at(row, price, book_side)
        delta = pre_sz - post_sz

        results.append({'idx': i, 'trade_price': price,
                        'agg_side': agg_side, 'book_side': book_side,
                        'sum_T_size': total_t_size, 'book_delta': delta,
                        'covers_trades': delta >= total_t_size})
    res = pd.DataFrame(results)
    print("\n=== Claim 3a: side='N' non-trade following an unterminated T-run ===")
    print(f"  cases examined: {len(res)}")
    if len(res):
        ok = res['covers_trades'].sum()
        print(f"  book delta >= sum(T sizes): {ok}/{len(res)} "
              f"({ok / len(res) * 100:.1f}%)")
        if (~res['covers_trades']).any():
            print("  first counterexamples:")
            print(res[~res['covers_trades']].head())
    return res

def validate_claim_3b(df, sessions):
    """
    Claim 3b: A non-trade side='N' row with NO pending T-run is invalid,
    operationalized as 'book unchanged vs. previous row'.
    """
    results = []
    for start, end, kind in tqdm(sessions, desc="Claim 3b", leave=False):
        if kind != 'single':
            continue
        i = start
        row = df.iloc[i]
        if row['side'] != 'N' or row['action'] == 'T':
            continue
        if has_pending_trade(df, i):
            continue
        if i == 0:
            continue
        unchanged = book_tuple(df.iloc[i - 1]) == book_tuple(row)
        results.append({'idx': i, 'action': row['action'],
                        'book_unchanged': unchanged})
    res = pd.DataFrame(results)
    print("\n=== Claim 3b: side='N' non-trade with no pending T-run ===")
    print(f"  cases examined: {len(res)}")
    if len(res):
        ok = res['book_unchanged'].sum()
        print(f"  book unchanged (invalid as predicted): {ok}/{len(res)} "
              f"({ok / len(res) * 100:.1f}%)")
        if (~res['book_unchanged']).any():
            print("  first counterexamples (book did change):")
            print(res[~res['book_unchanged']].head())
    return res

# ---------- driver ----------

def validate_all(df):
    df = df.reset_index(drop=True)
    sessions = find_sessions(df)
    print(f"Detected {len(sessions)} sessions in {len(df)} rows\n")
    # c1 = validate_claim_1(df, sessions)
    # c2 = validate_claim_2(df, sessions)
    # c3a = validate_claim_3a(df, sessions)
    c3b = validate_claim_3b(df, sessions)
    

df = pd.read_csv("data/raw/PFE/XNAS-20260425-UEJBDCM7RR/xnas-itch-20250401-20250430.mbp-10.PFE.csv")
results = validate_all(df)