#env variable management
from pydantic_settings import BaseSettings, SettingsConfigDict
#data type enforcement
from pydantic import RedisDsn, Field

class Settings(BaseSettings):
    
    #environment variable management
    #validating data types into python objects
    
    #Project INFO:
    PROJECT_NAME = "Echoshield"
    DEBUG: bool = False
    
    #configuring multiprocessing:
    #my mac has 8 cores of cpu
    WORKER_POOL_SIZE: int = Field(default=8, ge=1, le=16)
    
    #redis cache config:
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 3600  # Cache audio transcriptions for 1 hour
    
    # Pydantic v2 config for loading the .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    #model:
    MODEL_NAME = "whisper-tiny"
    
    #local cache config:
    LOCAL_CACHE_SIZE = "1024MB"
    LOCAL_CACHE_MAX_SIZE = "1024MB"
    LOCAL_CACHE_TTL_SECONDS = 3600 
    
    #device config:
    DEVICE = "mps" #macOS gpu hardware acceleration
    COMPUTE_TYPE = "float32"
    
    #sampling rate:
    SAMPLING_RATE = 16000
    
settings = Settings()
