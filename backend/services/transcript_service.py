import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

class InvalidYouTubeURL(Exception):
    pass


class TranscriptUnavailable(Exception):
    pass


class TranscriptService:

    def __init__(self):
        self.youtube_api = YouTubeTranscriptApi()

    def extract_video_id(self, url: str) -> str:
        """Extract and validate YouTube video ID."""

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
        """Fetch transcript and preserve timestamps."""

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
