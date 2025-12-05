#!/usr/bin/env python3
# scripts/generate_test_data.py
"""
Generate realistic synthetic futures data for testing.
FIXED: No duplicates, proper DatetimeIndex, enough samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)

MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

def generate_clean_data(symbol='CL', start_year=2015, end_year=2024, n_contracts=8):
    """Generate clean synthetic data WITHOUT duplicates."""
    # Generate business days
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    dates = pd.bdate_range(start_date, end_date).date
    
    log.info(f"Generating {len(dates)} business days for {symbol}")
    
    # Create base price path
    n_days = len(dates)
    returns = np.random.normal(0, 0.02/np.sqrt(252), n_days)
    price_path = 50.0 * np.exp(np.cumsum(returns))
    
    # Generate 8 consecutive contracts
    all_data = []
    contract_counter = 0
    
    for i, current_date in enumerate(dates):
        base_price = price_path[i]
        
        # Create term structure for this day
        for offset in range(n_contracts):
            # Expiry is 20th of month, N months out
            months_ahead = offset + 1
            expiry_year = current_date.year
            expiry_month = ((current_date.month - 1 + months_ahead) % 12) + 1
            expiry_year += (current_date.month - 1 + months_ahead) // 12
            expiry = date(expiry_year, expiry_month, 20)
            
            # Skip if expiry already passed
            if expiry <= current_date:
                continue
                
            # Price with contango
            price = base_price * (1 + 0.05 * offset + np.random.normal(0, 0.05))
            
            # Volume/OI patterns
            days_to_expiry = (expiry - current_date).days
            volume = max(100, int(10000 * np.exp(-days_to_expiry/100)))
            oi = max(1000, int(50000 * np.exp(-days_to_expiry/150)))
            
            # Front months more liquid
            if offset < 3:
                volume *= 3
                oi *= 2
            
            all_data.append({
                'date': current_date,
                'symbol': symbol,
                'expiry': expiry,
                'settle': round(price, 2),
                'last': round(price * np.random.uniform(0.999, 1.001), 2),
                'bid': round(price - 0.01, 2),
                'ask': round(price + 0.01, 2),
                'volume': int(volume),
                'open_interest': int(oi)
            })
    
    df = pd.DataFrame(all_data)
    
    # Remove any duplicates (shouldn't be any, but safe)
    df = df.drop_duplicates(subset=['date', 'expiry'])
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate clean synthetic futures data")
    parser.add_argument("--symbol", default="CL")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--contracts", type=int, default=8)
    
    args = parser.parse_args()
    configure_logging(level="INFO")
    
    symbol = args.symbol.upper()
    
    # Generate data
    df = generate_clean_data(
        symbol=symbol,
        start_year=args.start_year,
        end_year=args.end_year,
        n_contracts=args.contracts
    )
    
    # Save
    output_dir = ensure_dir(ROOT / "data" / "raw" / symbol)
    
    # Save by contract for realism
    contracts = df['expiry'].unique()
    for expiry in contracts[:12]:  # First 12 contracts only
        contract_df = df[df['expiry'] == expiry].copy()
        if not contract_df.empty:
            month_code = MONTH_CODES[expiry.month - 1]
            year_short = str(expiry.year)[-2:]
            filename = f"{symbol}{month_code}{year_short}.csv"
            contract_df.to_csv(output_dir / filename, index=False)
    
    # Also save combined
    df.to_csv(output_dir / f"{symbol}_all.csv", index=False)
    
    log.info(f"Generated {len(df)} records for {symbol}")
    log.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    log.info(f"Unique contracts: {len(contracts)}")
    log.info(f"No duplicates: {len(df) == len(df.drop_duplicates(subset=['date', 'expiry']))}")

if __name__ == "__main__":
    main()
