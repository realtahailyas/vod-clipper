"""One-time YouTube OAuth: creates a refresh-token file a posting agent can reuse.

Usage:
    python tools/youtube_auth.py secrets/client_secret.json secrets/youtube_token.json

Prints a Google consent URL; open it, pick the channel's brand account, allow.
The token file holds a refresh token — keep it out of git (secrets/ is ignored).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",     # upload videos
    "https://www.googleapis.com/auth/youtube.force-ssl",  # captions, playlists, metadata edits
    "https://www.googleapis.com/auth/youtube.readonly",   # read channel / video status
]


def main() -> int:
    if len(sys.argv) > 3 or any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 2
    client_secret = Path(sys.argv[1] if len(sys.argv) > 1
                         else os.environ.get("YOUTUBE_CLIENT_SECRET", "secrets/client_secret.json"))
    token_out = Path(sys.argv[2] if len(sys.argv) > 2
                     else os.environ.get("YOUTUBE_TOKEN", "secrets/youtube_token.json"))
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    creds = flow.run_local_server(
        host="localhost", port=8765, open_browser=False, prompt="consent",
        access_type="offline",
        authorization_prompt_message="AUTH_URL: {url}",
        success_message="Authorized. You can close this tab.",
    )
    token_out.parent.mkdir(parents=True, exist_ok=True)
    token_out.write_text(creds.to_json(), encoding="utf-8")
    print(f"TOKEN_SAVED: {token_out} (refresh token: {'yes' if creds.refresh_token else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
