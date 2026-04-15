from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs
from deep_translator import GoogleTranslator

def translate_to_english(text: str) -> str:
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception:
        return text  # fallback if translation fails
    
def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()

    try:
        # Get all available transcripts
        transcript_list = api.list(video_id)

        # Try English first
        try:
            transcript = transcript_list.find_transcript(["en"])
            lang = "en"
        except:
            # Fallback to any available transcript (like Hindi auto-generated)
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            )
            lang = transcript.language_code

        data = transcript.fetch()
        text = " ".join([chunk.text for chunk in data])

        # Translate if not English
        if lang != "en":
            text = translate_to_english(text)

        return text

    except Exception as e:
        raise Exception(f"Transcript error: {str(e)}")






def extract_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from URL or return ID if already provided.
    """

    # Case 1: Already an ID
    if len(url_or_id) == 11 and " " not in url_or_id:
        return url_or_id

    # Parse URL
    parsed_url = urlparse(url_or_id)

    # Case 2: Standard YouTube URL
    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        if "v" in query:
            return query["v"][0]

    # Case 3: Short URL
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")

    # Fallback (regex as backup)
    match = re.search(r"([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL or ID")


