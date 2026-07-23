import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a local .env file if present (does not override
# variables that are already set in the real environment, e.g. in CI/CD
# or a hosting platform).
load_dotenv()


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # ---- Pinecone -------------------------------------------------
    pinecone_api_key: str = _get_env("PINECONE_API_KEY")
    pinecone_cloud: str = _get_env("PINECONE_CLOUD", "aws")
    pinecone_region: str = _get_env("PINECONE_REGION", "us-east-1")
    pinecone_index_name: str = _get_env("PINECONE_INDEX_NAME", "rag-histo-index")
    pinecone_metric: str = _get_env("PINECONE_METRIC", "cosine")

    # ---- Groq (LLM) -------------------------------------------------
    # Mandatory: LLM key must be Groq Cloud, NOT OpenAI.
    groq_api_key: str = _get_env("GROQ_API_KEY")
    groq_model: str = _get_env("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ---- Embeddings ---------------------------------------------------
    # Sentence-Transformers model used for both indexing and query embedding.
    embedding_model_name: str = _get_env(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimension: int = int(_get_env("EMBEDDING_DIMENSION", "384"))

    # ---- OCR ------------------------------------------------------
    tesseract_cmd: str = _get_env("TESSERACT_CMD", "")  # e.g. "/usr/bin/tesseract"
    ocr_min_chars_per_page: int = int(_get_env("OCR_MIN_CHARS_PER_PAGE", "20"))

    # ---- STT (speech to text) --------------------------------------
    whisper_model_size: str = _get_env("WHISPER_MODEL_SIZE", "base")

    # ---- Chunking defaults --------------------------------------------
    default_chunk_size: int = int(_get_env("DEFAULT_CHUNK_SIZE", "800"))
    default_chunk_overlap: int = int(_get_env("DEFAULT_CHUNK_OVERLAP", "120"))

    # ---- Retrieval defaults -------------------------------------------
    default_top_k: int = int(_get_env("DEFAULT_TOP_K", "5"))
    default_score_threshold: float = float(_get_env("DEFAULT_SCORE_THRESHOLD", "0.20"))

    # ---- Misc -----------------------------------------------------
    max_upload_mb: int = int(_get_env("MAX_UPLOAD_MB", "20"))
    log_path: str = _get_env("LOG_PATH", "logs/query_log.jsonl")


settings = Settings()


def validate_required_keys():
    """Returns a list of human-readable problems with the current config.
    Used by the UI to show actionable error messages instead of raw
    stack traces."""
    problems = []
    if not settings.pinecone_api_key:
        problems.append("PINECONE_API_KEY is not set.")
    if not settings.groq_api_key:
        problems.append("GROQ_API_KEY is not set.")
    return problems
