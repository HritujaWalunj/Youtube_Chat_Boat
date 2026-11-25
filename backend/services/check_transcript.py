from transcript_service import TranscriptService

service = TranscriptService()

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