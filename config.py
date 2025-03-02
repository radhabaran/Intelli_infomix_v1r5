# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
api_key = os.environ['OA_API']
os.environ['OPENAI_API_KEY'] = api_key

# Huggingface API configuration
hf_api_key = os.environ['MY_HF_TOKEN1']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api_key

# Anthropic API configuration
anthro_api_key = os.environ['ANTHRO_KEY']
os.environ['ANTHROPIC_API_KEY'] = anthro_api_key

class Config:
    # Local server settings
    # QDRANT_HOST = "localhost"
    # QDRANT_PORT = 6333
    # Qdrant Cloud Settings (replace local settings)
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

    # API Key for OpenAI
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    # API Key for Huggingface
    HF_API_KEY = os.environ['HUGGINGFACEHUB_API_TOKEN']

    # API Key for Anthropic
    ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

    # Confluence settings
    CONFLUENCE_USERNAME = 'radhabaran.mohanty@gmail.com'
    CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_KEY')  # Load from environment variable
    CONFLUENCE_PAGE_ID = '98319'  # The ID of the page you want to access
    CONFLUENCE_BASE_URL = "https://radhatrial.atlassian.net/wiki"

    # Collection settings
    COLLECTION_NAME = "knowledge_base"
    COLLECTION_VERSION = "1.0"
    VECTOR_DIMENSION = 1536  # OpenAI ada-002 embedding dimension

    # File paths and directories
    BASE_DIR = Path("./data")
    DOCUMENT_DIRECTORY = BASE_DIR / "documents"
    LOCAL_QDRANT_PATH = BASE_DIR / "local_qdrant"
    PDF_DIRECTORY = DOCUMENT_DIRECTORY / "pdfs"
    PPT_DIRECTORY = DOCUMENT_DIRECTORY / "presentations"
    EXCEL_DIRECTORY = DOCUMENT_DIRECTORY / "excel"
    LOG_DIRECTORY = BASE_DIR / "logs"
    CACHE_DIRECTORY = BASE_DIR / "cache"
    TEMP_DIRECTORY = BASE_DIR / "temp"

    # Image storage settings
    IMAGE_STORAGE_PATH = BASE_DIR / "stored_images"
    IMAGE_STORAGE_FORMAT = "PNG"
    IMAGE_QUALITY = 100  # For JPEG images

    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.pdf', '.ppt', '.pptx', '.xlsx']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    # Search settings
    SEARCH_LIMIT = 30
    SIMILARITY_THRESHOLD = 0.83
    SCORE_THRESHOLD_EXCEL = 0.75
    MAX_SEARCH_RESULTS = 20
    RERANK_TOP_K = 5

    # Chunking configuration
    CHUNK_SIZE = 512  # Adjust as needed
    CHUNK_OVERLAP = 50  # Adjust as needed
    BATCH_SIZE = 100  # Adjust as needed
    MAX_TOKENS_PER_CHUNK = 2048
    MIN_CHUNK_LENGTH = 50

    # Image processing settings
    MAX_IMAGE_SIZE = 2048  # pixels
    MIN_IMAGE_SIZE = 100  # pixels
    TARGET_IMAGE_DPI = 600
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # Model settings
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    CLIP_PROCESSOR = "openai/clip-vit-base-patch32"
    GPT_MODEL = "gpt-4"

    # Performance settings
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT = 30  # seconds
    RATE_LIMIT_REQUESTS = 50
    RATE_LIMIT_PERIOD = 60  # seconds

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOG_DIRECTORY / "document_processor.log"
    ERROR_LOG_FILE = LOG_DIRECTORY / "errors.log"

    # Cache settings
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB

    INDEXING_THRESHOLD = 20000
    MAX_RETRIES = 5
    WAIT_TIME = 2
