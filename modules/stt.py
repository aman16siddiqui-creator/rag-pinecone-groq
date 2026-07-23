from __future__ import annotations

from functools import lru_cache

from config import settings


class STTError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_whisper_model():
    from faster_whisper import WhisperModel  # imported lazily: heavy dependency

    return WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe a recorded/uploaded audio file (wav/mp3/m4a/ogg) into
    text using a local Whisper model. Returns an empty string (rather
    than raising) on silence, so the UI can just say 'no speech detected'."""
    try:
        model = _get_whisper_model()
        segments, _info = model.transcribe(audio_file_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()
    except Exception as exc:
        raise STTError(f"Speech-to-text transcription failed: {exc}") from exc
