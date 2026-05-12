"""
approve_builder.py — register/approve a builder code with HL.

Run ONCE per wallet to authorize a builder address. After approval, every
order with builder={"b": <addr>, "f": <fee>} routes fees to the builder.

Usage:
    HL_PRIVATE_KEY=0x... \
    HL_BUILDER_ADDRESS=0x... \
    HL_BUILDER_FEE_TENTHS_BPS=10 \
    CONFIRM=yes \
        python3 scripts/approve_builder.py

After approval persists onchain, the engine BUILDER_ADDRESS env on each
engine is what gets passed on every order.

For self-rebate loop:
  HL_BUILDER_ADDRESS = same wallet as HL_ADDRESS → fees route to self
  HL pays a portion of collected taker fees back to builder (us)
"""
import os
import sys

PRIVATE_KEY = os.environ.get("HL_PRIVATE_KEY", "").strip()
BUILDER_ADDR = os.environ.get("HL_BUILDER_ADDRESS", "").strip().lower()
FEE_TENTHS_BPS = int(os.environ.get("HL_BUILDER_FEE_TENTHS_BPS", "10"))
TESTNET = os.environ.get("HL_TESTNET", "0") == "1"

if not PRIVATE_KEY or not BUILDER_ADDR:
    print("ERR: HL_PRIVATE_KEY and HL_BUILDER_ADDRESS env vars required",
          file=sys.stderr)
    sys.exit(1)

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

wallet = Account.from_key(PRIVATE_KEY)
api_url = constants.TESTNET_API_URL if TESTNET else constants.MAINNET_API_URL
exchange = Exchange(wallet, api_url)

fee_pct_str = f"{FEE_TENTHS_BPS / 100:.4f}%"

print(f"approving builder {BUILDER_ADDR} for wallet {wallet.address}")
print(f"max fee rate: {fee_pct_str}  ({FEE_TENTHS_BPS} tenths_bps = {FEE_TENTHS_BPS/10}bp)")
print(f"network: {'testnet' if TESTNET else 'MAINNET'}")
print()

if os.environ.get("CONFIRM") != "yes":
    print("DRY RUN — set CONFIRM=yes to actually submit the approval")
    sys.exit(0)

result = exchange.approve_builder_fee(BUILDER_ADDR, fee_pct_str)
print(f"\nresult: {result}")
