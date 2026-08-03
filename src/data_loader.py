"""
data_loader.py

Loads historical price data, aligns all assets by trading date,
and computes simple daily returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    FILE_MAP,
    DATE_COLUMN,
    PRICE_COLUMN,
)


def load_price_data():
    """
    Returns
    -------
    prices : pandas.DataFrame
        Daily closing prices aligned across all assets.
    """

    merged = None

    for asset, filename in FILE_MAP.items():

        filepath = DATA_DIR / filename

        df = pd.read_csv(filepath, encoding="utf-8-sig")

        # remove whitespace and BOM characters
        df.columns = [
            c.strip().replace("\ufeff", "")
            for c in df.columns
        ]

        if DATE_COLUMN not in df.columns:
            raise ValueError(f"{filename}: Missing '{DATE_COLUMN}' column.")

        if PRICE_COLUMN not in df.columns:
            raise ValueError(f"{filename}: Missing '{PRICE_COLUMN}' column.")

        df = df[[DATE_COLUMN, PRICE_COLUMN]]

        df.columns = [DATE_COLUMN, asset]

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

        df[asset] = (
            df[asset]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        df = df.sort_values(DATE_COLUMN)

        df = df.drop_duplicates(subset=DATE_COLUMN)

        df = df.set_index(DATE_COLUMN)

        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="inner")

    merged = merged.dropna()

    return merged


def compute_simple_returns(price_df):
    """
    Computes simple returns.

    Parameters
    ----------
    price_df : pandas.DataFrame

    Returns
    -------
    returns : pandas.DataFrame
    """

    returns = price_df.pct_change()

    returns = returns.dropna()

    return returns


def load_returns():
    """
    Complete data-loading pipeline.

    Returns
    -------
    R : ndarray
        T × N return matrix

    asset_names : list

    dates : DatetimeIndex
    """

    prices = load_price_data()

    returns = compute_simple_returns(prices)

    R = returns.values

    asset_names = list(returns.columns)

    dates = returns.index

    return R, asset_names, dates
