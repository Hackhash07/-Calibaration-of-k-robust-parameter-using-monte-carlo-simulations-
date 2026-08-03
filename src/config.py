"""
config.py

Central configuration file for the Robust Portfolio Optimization project.

Modify only this file when changing:
    - data location
    - experiment parameters
    - optimization settings
"""

from pathlib import Path
import numpy as np

# ============================================================
# Project directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "output"

FIGURE_DIR = OUTPUT_DIR / "figures"
RETURN_DIR = OUTPUT_DIR / "returns"
WEIGHT_DIR = OUTPUT_DIR / "weights"
LOG_DIR = OUTPUT_DIR / "logs"

# Create output folders if they do not exist
for folder in [
    FIGURE_DIR,
    RETURN_DIR,
    WEIGHT_DIR,
    LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# Asset universe
# ============================================================

FILE_MAP = {
    "Reliance": "RELIANCE.csv",
    "TCS": "TCS.csv",
    "HDFCBank": "HDFCBANK.csv",
    "Infosys": "INFY.csv",
    "ICICIBank": "ICICIBANK.csv",
    "HindustanUL": "HINDUNILVR.csv",
    "ITC": "ITC.csv",
    "SBI": "SBIN.csv",
    "BajajFin": "BAJFINANCE.csv",
    "Airtel": "BHARTIARTL.csv",
    "KotakBank": "KOTAKBANK.csv",
    "AxisBank": "AXISBANK.csv",
    "AsianPaints": "ASIANPAINT.csv",
    "Maruti": "MARUTI.csv",
    "Wipro": "WIPRO.csv",
    "HCLTech": "HCLTECH.csv",
    "LT": "LT.csv",
    "MahindraM": "M&M.csv",
    "Titan": "TITAN.csv",
    "Nestle": "NESTLEIND.csv",
    "UltraTech": "ULTRACEMCO.csv",
    "PowerGrid": "POWERGRID.csv",
    "NTPC": "NTPC.csv",
    "SunPharma": "SUNPHARMA.csv",
    "DrReddy": "DRREDDY.csv",
    "AdaniPorts": "ADANIPORTS.csv",
    "TataMotors": "TATAMOTORS.csv",
    "TataSteel": "TATASTEEL.csv",
    "JSWSteel": "JSWSTEEL.csv",
    "DivisLabs": "DIVISLAB.csv",
}

# ============================================================
# Data format
# ============================================================

DATE_COLUMN = "Date"

PRICE_COLUMN = "ClosePrice"

# ============================================================
# Optimization parameters
# ============================================================

LAMBDA = 1.0

LONG_ONLY = True

FULL_INVESTMENT = False

# ============================================================
# Kappa sweep
# ============================================================

N_ALPHA = 21

ALPHA_GRID = np.linspace(
    0.0,
    1.0,
    N_ALPHA,
)

# ============================================================
# Numerical solver
# ============================================================

SOLVER = "CLARABEL"

# ============================================================
# Random seed
# ============================================================

RANDOM_SEED = 42
