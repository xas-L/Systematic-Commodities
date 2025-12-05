#!/usr/bin/env python3
# scripts/generate_test_data.py
"""
Generate realistic synthetic futures data for testing curve construction and walk-forward backtests.
Creates individual contract CSV files in data/raw/<SYMBOL>/ with realistic term structure patterns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)

# Constants
MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
MONTH_CODE_TO_NUM = {code: i+1 for i, code in enumerate(MONTH_CODES)}

def generate_business_days(start: date, end: date) -> List[date]:
    """Generate business days (Monday-Friday) between dates."""
    all_days = pd.bdate_range(start=start, end=end).date
    return [d for d in all_days]

def calculate_expiry_date(year: int, month_code: str) -> date:
    """Calculate approximate expiry date (20th of month for most commodities)."""
    month = MONTH_CODE_TO_NUM[month_code]
    # For testing: expiry on 20th of month, or preceding business day
    expiry = date(year, month, 20)
    # Adjust to business day
    while expiry.weekday() >= 5:  # Saturday or Sunday
        expiry -= timedelta(days=1)
    return expiry

def generate_term_structure(
    base_price: float,
    days_to_expiry: int,
    total_contracts: int = 12,
    contango_factor: float = 0.0005,
    seasonality_amplitude: float = 2.0
) -> np.ndarray:
    """
    Generate realistic term structure with:
    - Contango (far months > near months)
    - Seasonality (certain months higher/lower)
    - Time decay as expiry approaches
    """
    # Base contango curve
    curve = np.ones(total_contracts)
    for i in range(total_contracts):
        curve[i] = 1.0 + contango_factor * i
    
    # Add seasonality (e.g., winter premium for NG, summer for gasoline)
    month_idx = (np.arange(total_contracts) % 12)
    # Winter months (Oct-Mar: higher) for energy
    seasonality = np.where(
        np.isin(month_idx, [9, 10, 11, 0, 1, 2]),  # Oct-Mar
        seasonality_amplitude * 0.01,
        -seasonality_amplitude * 0.005
    )
    curve += seasonality
    
    # Time decay effect for near months
    if days_to_expiry < 30:
        # Near expiry contracts might have different structure
        near_months = min(3, total_contracts)
        decay = np.exp(-0.05 * (30 - days_to_expiry) / 30)
        curve[:near_months] *= decay
    
    # Apply to base price with noise
    prices = base_price * curve * np.random.uniform(0.99, 1.01, total_contracts)
    return prices

def generate_contract_data(
    symbol: str,
    start_date: date,
    end_date: date,
    total_contracts: int = 12,
    base_price: float = 50.0,
    volatility: float = 0.02
) -> Dict[Tuple[int, str], pd.DataFrame]:
    """
    Generate synthetic data for all contracts.
    
    Returns dict mapping (year, month_code) -> DataFrame
    """
    # Generate all business days
    business_days = generate_business_days(start_date, end_date)
    log.info(f"Generating {len(business_days)} business days from {start_date} to {end_date}")
    
    # Generate contract specifications
    contracts = []
    years = range(start_date.year, end_date.year + 2)  # Extra year for forward curve
    for year in years:
        for month_code in MONTH_CODES:
            expiry = calculate_expiry_date(year, month_code)
            if expiry > start_date:  # Only include contracts that exist during period
                contracts.append({
                    'year': year,
                    'month_code': month_code,
                    'expiry': expiry,
                    'contract_symbol': f"{symbol}{month_code}{str(year)[-2:]}"
                })
    
    # Container for all contract data
    all_data = {}
    
    # Generate price path with realistic features
    returns = np.random.normal(0, volatility / np.sqrt(252), len(business_days))
    # Add drift and mean reversion
    price_path = base_price * np.exp(np.cumsum(returns))
    # Add some trends and cycles
    t = np.arange(len(business_days)) / 252.0  # Time in years
    price_path *= (1 + 0.1 * np.sin(2 * np.pi * t) + 0.05 * np.sin(4 * np.pi * t))
    
    for day_idx, current_date in enumerate(business_days):
        current_price = price_path[day_idx]
        
        # For each day, determine which contracts are active
        active_contracts = [
            c for c in contracts 
            if c['expiry'] > current_date  # Not expired
            and (c['expiry'] - current_date).days > 5  # Not too close to expiry
        ]
        
        # Sort by expiry
        active_contracts.sort(key=lambda x: x['expiry'])
        
        # Take first N contracts for curve
        if len(active_contracts) > total_contracts:
            active_contracts = active_contracts[:total_contracts]
        
        # Generate term structure for this day
        days_to_front = (active_contracts[0]['expiry'] - current_date).days
        term_prices = generate_term_structure(
            current_price,
            days_to_front,
            total_contracts=len(active_contracts)
        )
        
        # Add data for each active contract
        for contract_idx, contract in enumerate(active_contracts):
            key = (contract['year'], contract['month_code'])
            
            if key not in all_data:
                all_data[key] = {
                    'date': [], 'symbol': [], 'expiry': [], 'settle': [],
                    'last': [], 'bid': [], 'ask': [], 'volume': [], 'open_interest': []
                }
            
            settle_price = term_prices[contract_idx]
            
            # Realistic volume and open interest patterns
            days_to_expiry = (contract['expiry'] - current_date).days
            volume_base = 10000
            oi_base = 50000
            
            # Volume/OI decay as expiry approaches
            if days_to_expiry < 60:
                volume_factor = max(0.1, days_to_expiry / 60)
                oi_factor = max(0.2, days_to_expiry / 60)
            else:
                volume_factor = 1.0
                oi_factor = 1.0
            
            volume = int(volume_base * volume_factor * np.random.uniform(0.8, 1.2))
            open_interest = int(oi_base * oi_factor * np.random.uniform(0.9, 1.1))
            
            # Front months have higher volume/OI
            if contract_idx < 3:
                volume *= np.random.uniform(2.0, 3.0)
                open_interest *= np.random.uniform(1.5, 2.0)
            
            # Add some noise to other fields
            last_price = settle_price * np.random.uniform(0.999, 1.001)
            spread = settle_price * 0.0005  # 5 bps spread
            bid_price = settle_price - spread/2
            ask_price = settle_price + spread/2
            
            all_data[key]['date'].append(current_date)
            all_data[key]['symbol'].append(symbol)
            all_data[key]['expiry'].append(contract['expiry'])
            all_data[key]['settle'].append(round(settle_price, 2))
            all_data[key]['last'].append(round(last_price, 2))
            all_data[key]['bid'].append(round(bid_price, 2))
            all_data[key]['ask'].append(round(ask_price, 2))
            all_data[key]['volume'].append(volume)
            all_data[key]['open_interest'].append(open_interest)
    
    # Convert to DataFrames
    result = {}
    for key, data_dict in all_data.items():
        if data_dict['date']:  # Only if we have data
            df = pd.DataFrame(data_dict)
            df = df.sort_values('date')
            result[key] = df
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic futures data for testing"
    )
    parser.add_argument("--symbol", default="CL", help="Root symbol (e.g., CL, NG, ZC)")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--contracts", type=int, default=12, help="Number of contracts in curve")
    parser.add_argument("--base-price", type=float, default=50.0, help="Base price for front month")
    parser.add_argument("--volatility", type=float, default=0.02, help="Annual volatility")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level="INFO")
    
    symbol = args.symbol.upper()
    start_date = date(args.start_year, 1, 1)
    end_date = date(args.end_year, 12, 31)
    
    # Generate data
    log.info(f"Generating synthetic data for {symbol} from {start_date} to {end_date}")
    contracts_data = generate_contract_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        total_contracts=args.contracts,
        base_price=args.base_price,
        volatility=args.volatility
    )
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "data" / "raw" / symbol
    
    ensure_dir(output_dir)
    
    # Write CSV files
    files_written = 0
    for (year, month_code), df in contracts_data.items():
        if not df.empty:
            filename = f"{symbol}{month_code}{str(year)[-2:]}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            files_written += 1
            log.debug(f"Written {filename} with {len(df)} records")
    
    # Also write a combined file for convenience
    if contracts_data:
        all_data = pd.concat(list(contracts_data.values()), ignore_index=True)
        all_data = all_data.sort_values(['date', 'expiry'])
        combined_path = output_dir / f"{symbol}_all.csv"
        all_data.to_csv(combined_path, index=False)
        log.info(f"Written combined file: {combined_path}")
    
    log.info(f"Generated {files_written} contract files for {symbol}")
    log.info(f"Data saved to: {output_dir}")
    
    # Print summary
    if contracts_data:
        first_df = list(contracts_data.values())[0]
        log.info(f"Date range: {first_df['date'].min()} to {first_df['date'].max()}")
        log.info(f"Contracts per day: ~{len(contracts_data)}")
        log.info(f"Total records: {len(all_data) if contracts_data else 0}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())