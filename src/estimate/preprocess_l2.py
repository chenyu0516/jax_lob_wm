import pandas as pd
from datetime import datetime
import numpy as np
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

import pandas_market_calendars as mcal

from dataclasses import dataclass
from typing import Literal

N_LEVELS = 4

@dataclass
class BookChange:
    side:        Literal['bid', 'ask']
    level:       int   # 0-indexed position in the book
    price:       int
    size_delta:  int   # positive = added, negative = removed

def parse_side(row, side: str, n_levels: int) -> dict[int, tuple[int, int]]:
    """Extract {price: (level, size)} for one side, skipping empty levels."""
    best = row[f'{side}_px_00']
            
    return {
        int(row[f'{side}_px_{i:02d}']): (int(abs(row[f'{side}_px_{i:02d}']-best)*1e-7)+1, int(row[f'{side}_sz_{i:02d}']))
        for i in range(n_levels)
        if abs(row[f'{side}_px_{i:02d}']-best) < (n_levels-0.5)*1e7 
        # only consider the change in first four queue level (each level is a tick away from the previous one)
        # minus 0.5 to prevent python overflow
    }

def book_diff(old_row, new_row, n_levels: int = N_LEVELS) -> list[BookChange]:
    """Detecting the difference between two snapshots, N_LEVELS can help you focus on price level in N_LEVELS*ticksize"""
    changes = []

    for side in ('bid', 'ask'):
        old_book = parse_side(old_row, side, n_levels)
        new_book = parse_side(new_row, side, n_levels)

        for price in old_book.keys() | new_book.keys():
            old_level, old_size = old_book.get(price, (None, 0))
            new_level, new_size = new_book.get(price, (None, 0))

            if old_size != new_size:
                # Prefer the new level; fall back to old if price was removed
                level = new_level if new_level is not None else old_level
                changes.append(BookChange(side, level, price, new_size - old_size))

    return changes



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



def get_imbalance_bin(series):
    """Discretelize the imbalance"""
    v = series.to_numpy()
    
    # Divide by 0.1 and round to fix floating-point precision issues
    # e.g., -0.1 / 0.1 could be -0.9999... instead of -1.0
    scaled = np.round(v / 0.1, 8)

    result = np.where(
        v == 0,                          # bin 0: exactly zero
        0,
        np.where(
            v < 0,
            np.floor(scaled).astype(int),  # negative: [0.1*i, 0.1*(i+1))
            np.ceil(scaled).astype(int)    # positive: (0.1*(i-1), 0.1*i]
        )
    )
    return result



def single_file_processor(dir: str):
    df = pd.read_csv(dir)

    previous = None
    events = []  # collect dicts, concat once at the end
    states = []

    for ts, temp_df in tqdm(df.groupby('ts_recv', sort=False), total=df['ts_recv'].nunique(), desc='Processing'):

        now = temp_df.iloc[-1].to_dict()
        
        ask = parse_side(now, 'ask', n_levels=N_LEVELS)
        bid = parse_side(now, 'bid', n_levels=N_LEVELS)
        
        best_ask = min(ask.keys())
        best_bid = max(bid.keys())
        
        if previous is not None:
            changes = book_diff(previous, now)        
            if changes:  # skip empty diffs
                reduce_reason = 'T' if (temp_df['action'] == 'T').any() else 'C'
                
                is_create = (best_ask < best_ask_p) or (best_bid > best_bid_p)
                
                for c in changes:
                    sd = c.size_delta
                    action = ('E' if (is_create and c.level == 1) else 'A') if sd > 0 else reduce_reason
                    events.append({
                        'ts_hap': ts, 'ts_con': previous['ts_recv'],
                        'side': c.side, 'level': c.level, 'price': c.price,
                        'size_delta': abs(sd), 'action': action,
                    })

                levels = range(-N_LEVELS, N_LEVELS+1)
                state = {lv: 0 for lv in levels}
                state.update({level: size for level, size in ask.values()})
                state.update({-level: size for level, size in bid.values()})

                bid1, ask1 = state[-1], state[1]
                denom = ask1 + bid1
                state['spread'] = int((best_ask - best_bid) * 1e-7)
                state['imb'] = (ask1 - bid1) / denom if denom != 0 else 2
                state['best_px'] = (best_ask + best_bid) / 2 * 1e-9
                states.append(state)

            
        previous = now  # always update, even on first iter
        best_ask_p = best_ask 
        best_bid_p = best_bid
        
    event_df = pd.DataFrame(events, columns=['ts_hap', 'ts_con', 'side', 'level', 'price', 'size_delta', 'action'])
    states_df = pd.DataFrame(states)
    states_df['ts'] = df['ts_recv'].drop_duplicates(ignore_index=True)
    states_df.drop(columns=[0], inplace=True)
        
    states_df['imb'] = get_imbalance_bin(states_df['imb'])
    states_df = filter_trading_hours(states_df, 'ts')
    event_df = filter_trading_hours(event_df, 'ts')
    
    return states_df, event_df


def batch_process(in_dir: str, out_dir: str):
    """Do all the data preprocessing for every .csv file in the directory"""
    directory = Path(in_dir)

    # 2. Find all CSV files (use .rglob('*.csv') to search subfolders too)
    csv_files = directory.glob('*.csv')
    sorted_files = sorted(csv_files, key=lambda f: f.name.split('-')[3])

    # 3. Read and combine into a single DataFrame
    states = []
    event = []
    
    batch_len = len(sorted_files)
    for i, f in enumerate(sorted_files):
        print(f"Batch file process: {i}/{batch_len}")
        
        states_df, event_df = single_file_processor(f)
        states.append(states_df)
        event.append(event_df)
        
    pd.concat(states, ignore_index=True).to_parquet(f'{out_dir}/states.parquet')
    pd.concat(event, ignore_index=True).to_parquet(f'{out_dir}/event.parquet')
    