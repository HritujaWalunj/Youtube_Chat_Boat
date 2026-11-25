"""Fetch a YouTube video's captions (and title).

Changes in this version
-----------------------
* NEW: ``fetch_title`` - looks up the video title so chats in the sidebar get
  a readable name. It is best-effort: on any failure the UI falls back to the
  video id, and loading the video is never blocked by it.
* The rest of the file is unchanged apart from added docstrings and comments.
"""

import json
import re
import urllib.request
from urllib.parse import parse_qs, quote, urlparse

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)


class InvalidYouTubeURL(Exception):
    """The text the user entered is not a recognisable YouTube video link."""


class TranscriptUnavailable(Exception):
    """The video has no usable English captions."""


class TranscriptService:
    """Turns a YouTube URL into a video id, a transcript and a title."""

    def __init__(self):
        self.youtube_api = YouTubeTranscriptApi()

    def extract_video_id(self, url: str) -> str:
        """Extract and validate YouTube video ID.

        Understands ``youtube.com/watch?v=``, ``/shorts/``, ``/embed/``,
        ``/live/`` and ``youtu.be/`` links. A valid id is exactly 11 characters
        (letters, digits, ``_`` or ``-``); anything else raises
        :class:`InvalidYouTubeURL`.
        """

        parsed_url = urlparse(url)

        hostname = parsed_url.hostname or ""
        hostname = hostname.lower()

        video_id = None

        if hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed_url.path == "/watch":
                video_id = parse_qs(parsed_url.query).get("v", [None])[0]

            elif parsed_url.path.startswith(("/shorts/", "/embed/", "/live/")):
                video_id = parsed_url.path.split("/")[2]

        elif hostname in ("youtu.be", "www.youtu.be"):
            video_id = parsed_url.path.strip("/").split("/")[0]

        if not video_id or not re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
            raise InvalidYouTubeURL("Please enter a valid YouTube video URL.")

        return video_id

    def fetch_transcript(self, url: str) -> dict:
        """Fetch transcript and preserve timestamps.

        Returns a dict with ``video_id``, the full ``transcript`` text, the raw
        ``segments`` (each ``{"text", "start", "duration"}``, used later to
        build timestamp-aware chunks) and the ``language``.
        """

        video_id = self.extract_video_id(url)

        try:
            transcript = self.youtube_api.fetch(
                video_id,
                languages=["en"]
            )

            segments = transcript.to_raw_data()

            if not segments:
                raise TranscriptUnavailable(
                    "No transcript content found for this video."
                )

            transcript_text = " ".join(
                segment["text"] for segment in segments
            )

            return {
                "video_id": video_id,
                "transcript": transcript_text,
                "segments": segments,
                "language": "en"
            }

        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            raise TranscriptUnavailable(
                "Sorry, English captions are not available for this video. "
                "Please try another video."
            ) from exc

    def fetch_title(self, video_id: str) -> str | None:
        """Return the video's title, or ``None`` if it cannot be determined.

        Uses YouTube's public oEmbed endpoint (no API key needed).
        """
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        endpoint = f"https://www.youtube.com/oembed?format=json&url={quote(watch_url, safe='')}"
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                return json.load(response).get("title")
        except Exception:
            return None  # cosmetic feature: never fail the request because of it
