"""Constants for the EmbeddingGemma 2 Core ML conversion pipeline."""

REPO_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SHORT_REVISION = REVISION[:8]

BUCKETS = (128, 256, 512)
FUNCTION_PREFIX = "len_"
DEFAULT_FUNCTION = "len_128"

EMBEDDING_DIM = 768
DOCUMENT_FORMAT = "title: {title} | text: {text}"
QUERY_FORMAT = "task: code retrieval | query: {query}"

MIN_ANE_OPS = 2360
MIN_COSINE = 0.9995

SAMPLES_PER_BUCKET = 4

# The fp16 ONNX build the off-macOS archive is rewritten from.
ONNX_SOURCE_REPO = "lexoliu/repo-search-embeddinggemma2-onnx"
ONNX_SOURCE_REVISION = "d4e1b374f97451cf89262d03f4f8e387822511d4"
ONNX_SOURCE_FILE = "eg2-onnx-fp16-914f7f89-ac32bf70.tar.zst"
ONNX_SOURCE_SHA256 = "ac32bf7023a81cf0d58e5f487ecb8e46979898d966f606bc564c5dba8585b414"
ONNX_SOURCE_MEMBERS = ("manifest.json", "model_fp16.onnx", "model_fp16.onnx_data", "tokenizer.json")

# The rewritten graph computes the source graph's values; only fp16
# rounding order may differ.
ONNX_MIN_COSINE = 0.9999