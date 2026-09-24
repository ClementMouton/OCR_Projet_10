import os
from dotenv import load_dotenv

load_dotenv()

# --- API Mistral ---

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print(
        "⚠️ Attention : la clé API Mistral "
        "(MISTRAL_API_KEY) n'est pas définie dans le fichier .env"
    )

MODEL_NAME = os.getenv("MODEL_ID", "mistral-small-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")


# --- Données et Vector Store ---

INPUT_DIR = os.getenv("INPUT_DIR", "inputs")

VECTOR_DB_DIR = os.getenv("OUTPUT_DIR", "vector_db")

FAISS_INDEX_FILE = os.getenv(
    "FAISS_INDEX_FILE",
    os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
)

DOCUMENT_CHUNKS_FILE = os.getenv(
    "DOCUMENT_CHUNKS_FILE",
    os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

SEARCH_K = int(os.getenv("SEARCH_K", "5"))


# --- Base de données ---

DATABASE_DIR = os.getenv("DATABASE_DIR", "database")

DATABASE_FILE = os.getenv(
    "DATABASE_FILE",
    os.path.join(DATABASE_DIR, "interactions.db")
)

DATABASE_URL = f"sqlite:///{DATABASE_FILE}"


# --- Application ---

APP_TITLE = os.getenv("APP_TITLE", "NBA Analyst AI")
NAME = os.getenv("NAME", "NBA")