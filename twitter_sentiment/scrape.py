from __future__ import annotations

import subprocess
import sys
import shutil
from typing import Dict, Generator, Iterable, Optional

from . import jsonio


def build_search_query(
    query: str,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    exclude_retweets: bool = True,
) -> str:
    """Build an snscrape search query string from parts.

    Dates must be YYYY-MM-DD if provided.
    """
    tokens: list[str] = [query]
    if lang:
        tokens.append(f"lang:{lang}")
    if exclude_retweets:
        tokens.append("exclude:retweets")
    if since:
        tokens.append(f"since:{since}")
    if until:
        tokens.append(f"until:{until}")
    # Ensure spaces between tokens; snscrape handles quoting internally when passed as a single arg
    return " ".join(token for token in tokens if token)


def normalize_tweet(raw: Dict) -> Dict:
    """Extract a compact, useful subset of fields from snscrape output."""
    user = raw.get("user") or {}
    return {
        "id": raw.get("id"),
        "url": raw.get("url"),
        "date": raw.get("date"),
        "content": raw.get("content"),
        "lang": raw.get("lang"),
        "username": user.get("username"),
        "displayname": user.get("displayname"),
        "replyCount": raw.get("replyCount"),
        "retweetCount": raw.get("retweetCount"),
        "likeCount": raw.get("likeCount"),
        "quoteCount": raw.get("quoteCount"),
        "conversationId": raw.get("conversationId"),
        "inReplyToTweetId": raw.get("inReplyToTweetId"),
        "mentionedUsers": raw.get("mentionedUsers"),
        "hashtags": raw.get("hashtags"),
    }


def run_snscrape_search(
    query: str,
    limit: int = 100,
    lang: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    exclude_retweets: bool = True,
    raw: bool = False,
) -> Generator[Dict, None, None]:
    """Yield tweets from snscrape `twitter-search` as dicts (normalized or raw)."""
    search_query = build_search_query(
        query=query,
        lang=lang,
        since=since,
        until=until,
        exclude_retweets=exclude_retweets,
    )
    snscrape_path = shutil.which("snscrape")
    if snscrape_path:
        cmd = [
            snscrape_path,
            "--jsonl",
            "--max-results",
            str(limit),
            "twitter-search",
            search_query,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "snscrape",
            "--jsonl",
            "--max-results",
            str(limit),
            "twitter-search",
            search_query,
        ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if not line.strip():
            continue
        try:
            obj = jsonio.loads(line)
        except Exception:
            # Skip lines that fail to parse; forward stderr to our stderr
            print(line, file=sys.stderr)
            continue
        yield obj if raw else normalize_tweet(obj)
    # Drain stderr to avoid zombies
    if proc.stderr is not None:
        proc.stderr.read()
    proc.wait()

