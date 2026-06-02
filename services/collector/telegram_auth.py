"""
telegram_auth.py
----------------
Interactive first-run authentication helper for Telegram / Telethon.

Run this script once inside the collector container to create the session file:

    docker compose run --rm collector python /app/telegram_auth.py

You will be prompted for:
  1. Your phone number (with country code, e.g. +1555123456)
  2. The OTP code sent to your Telegram app

After this completes a session file is saved under /app/data/telegram_sessions/
and subsequent non-interactive runs will use it automatically.

DO NOT commit the session file or your .env to source control.
"""
from __future__ import annotations

import os
import sys

TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "worldcup_telegram")
TELEGRAM_SESSION_DIR = "/app/data/telegram_sessions"


def main() -> None:
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        print(
            "ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from telethon.sync import TelegramClient  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: telethon is not installed. Rebuild the collector image first.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(TELEGRAM_SESSION_DIR, exist_ok=True)
    session_path = os.path.join(TELEGRAM_SESSION_DIR, TELEGRAM_SESSION_NAME)

    print(f"Authenticating Telethon session at: {session_path}")
    print("You will be asked for your phone number and a verification code.\n")

    with TelegramClient(session_path, int(TELEGRAM_API_ID), TELEGRAM_API_HASH) as client:
        client.start()  # triggers interactive phone/OTP prompts
        me = client.get_me()
        print(f"\nAuthenticated as: {getattr(me, 'username', None) or getattr(me, 'first_name', 'unknown')}")
        print("Session saved. You can now run the collector non-interactively.")


if __name__ == "__main__":
    main()
