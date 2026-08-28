import os
import io
from rag_engine.core.interfaces import TextToSpeechInterface, SpeechToTextInterface
from io import BytesIO
from gtts import gTTS
from google import genai
from google.genai import types

class TextToSpeechService(TextToSpeechInterface):
    def __init__(self):
        pass
    def synthesize(self, text:str) -> BytesIO:
        fp = BytesIO()
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(fp)
        fp.seek(0)

        return fp

class GeminiTTSService(SpeechToTextInterface):
    """Transcribes audio using Gemini's native multimodal capabilities."""
    def __init__(self, model_name:str="gemini-3.6-flash", key:str=None,):
        api_key = key or os.getenv("API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def transcribe(self, audio_bytes:io.BytesIO) -> str:
        try:
            audio_data = audio_bytes.read()
            prompt = (
                "Listen to the following audio and provide an exact, "
                "verbatim text transcription. Do not summarize or answer the audio, "
                "only output the spoken words."
            )
            audio_part = types.Part.from_bytes(
                data=audio_data,
                mime_type="audio/wav"
            )
            response = self.client.models.generate_content(
                model= self.model_name,
                contents=[prompt,audio_part]
            )

            return response.text.strip()
        except Exception as e :
            print(f"Gemini STT Error: {e}")
            return ""