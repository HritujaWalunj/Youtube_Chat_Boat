"""Manual smoke test for the transcript service (not used by the app).

Run from the project root:

    python -m backend.services.check_transcript

CHANGED: the import below used to be ``from transcript_service import ...``,
which only worked when the script was started from inside this folder. It now
uses the full package path so the command above works from the project root.
"""

from backend.services.transcript_service import TranscriptService

service = TranscriptService()

# Any video that has English captions.
url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"

try:
    result = service.fetch_transcript(url)

    print("Video ID:", result["video_id"])
    print("Language:", result["language"])
    print("Transcript length:", len(result["transcript"]))
    print("First 500 characters:")
    print(result["transcript"][:500])

except Exception as error:
    print("Error:", error)
