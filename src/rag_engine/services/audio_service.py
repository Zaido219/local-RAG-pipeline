from rag_engine.core.interfaces import TextToSpeechInterface, SpeechToText
from io import BytesIO
from gtts import gTTS

class TextToSpeechService(TextToSpeechInterface):
    def __init__(self):
        pass
    def synthesize(self, text:str) -> BytesIO:
        fp = BytesIO()
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(fp)
        fp.seek(0)

        return fp