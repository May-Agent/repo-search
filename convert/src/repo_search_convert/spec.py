"""Constants for the EmbeddingGemma 2 Core ML conversion pipeline."""

REPO_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SHORT_REVISION = REVISION[:8]

BUCKETS = (128, 256, 512, 1024)
FUNCTION_PREFIX = "len_"
DEFAULT_FUNCTION = "len_128"

EMBEDDING_DIM = 768
DOCUMENT_FORMAT = "title: {title} | text: {text}"
QUERY_FORMAT = "task: code retrieval | query: {query}"

MIN_ANE_OPS = 2360
MIN_COSINE = 0.9995

SAMPLES_PER_BUCKET = 4