"""Upload one packaged Short (a publish/<creator>/<NNN - ...>/ folder) to YouTube.

Usage:
    python tools/youtube_upload.py PACKAGE_DIR --token secrets/youtube_token.json
        [--privacy public|unlisted|private] [--publish-at 2026-11-20T17:00:00Z] [--dry-run]

Reads title.txt, description.txt, tags.txt, captions.srt and the .mp4 in the folder, uploads the
video (category Gaming, not made for kids), adds the English captions, and writes
status / video_id / url / uploaded_at back into the folder's metadata.json.
With --publish-at the video is uploaded private and YouTube publishes it at that UTC time.
Quota: 100 uploads/day per Google Cloud project.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

GAMING = "20"


def load_package(folder: Path) -> dict:
    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 in {folder}")
    read = lambda n: (folder / n).read_text(encoding="utf-8").strip() if (folder / n).exists() else ""
    tags = [t.strip() for t in read("tags.txt").split(",") if t.strip()]
    return {"video": videos[0], "title": read("title.txt")[:100], "description": read("description.txt")[:5000],
            "tags": tags, "captions": folder / "captions.srt" if (folder / "captions.srt").exists() else None,
            "meta_path": folder / "metadata.json"}


def youtube(token: Path):
    creds = Credentials.from_authorized_user_file(str(token))
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        token.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji in descriptions on Windows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--token", type=Path, default=Path(os.environ.get("YOUTUBE_TOKEN", "secrets/youtube_token.json")),
                    help="OAuth token file (default: $YOUTUBE_TOKEN or secrets/youtube_token.json)")
    ap.add_argument("--privacy", choices=["public", "unlisted", "private"], default="public")
    ap.add_argument("--publish-at", help="UTC ISO time; uploads private and schedules publishing")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    p = load_package(a.package)
    status = {"privacyStatus": "private" if a.publish_at else a.privacy,
              "selfDeclaredMadeForKids": False, "containsSyntheticMedia": False}
    if a.publish_at:
        status["publishAt"] = a.publish_at
    body = {"snippet": {"title": p["title"], "description": p["description"], "tags": p["tags"],
                        "categoryId": GAMING, "defaultLanguage": "en", "defaultAudioLanguage": "en"},
            "status": status}
    print(json.dumps({"file": str(p["video"]), **body}, indent=2, ensure_ascii=False))
    if a.dry_run:
        return 0

    yt = youtube(a.token)
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(p["video"]), chunksize=8 * 1024 * 1024,
                                                        resumable=True, mimetype="video/mp4"))
    resp = None
    while resp is None:
        prog, resp = req.next_chunk()
        if prog:
            print(f"uploading {prog.progress():.0%}", flush=True)
    vid = resp["id"]
    url = f"https://www.youtube.com/shorts/{vid}"
    print("UPLOADED:", url)

    if p["captions"]:
        try:
            yt.captions().insert(part="snippet", body={"snippet": {"videoId": vid, "language": "en",
                                                                   "name": "English", "isDraft": False}},
                                 media_body=MediaFileUpload(str(p["captions"]), mimetype="application/octet-stream")
                                 ).execute()
            print("captions added")
        except Exception as e:  # captions are a bonus; never fail the upload for them
            print("captions failed:", str(e).splitlines()[0][:200])

    if p["meta_path"].exists():
        meta = json.loads(p["meta_path"].read_text(encoding="utf-8"))
        meta.update({"status": "scheduled" if a.publish_at else "posted", "video_id": vid, "url": url,
                     "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        if a.publish_at:
            meta["scheduled_for"] = a.publish_at
        p["meta_path"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
