#env variable management
from pydantic_settings import BaseSettings, SettingsConfigDict
#data type enforcement
from pydantic import Field

class Settings(BaseSettings):
    
    #environment variable management
    #validating data types into python objects
    
    #Project INFO:
    PROJECT_NAME: str = "Echoshield"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    #configuring multiprocessing:
    #my mac has 8 cores of cpu
    POOL_WORKERS: int | None = Field(default=None, ge=1, le=16)
    
    #redis cache config:
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = Field(
        default=3600,
        validation_alias="CACHE_TTL_SECONDS",
    )  # Cache audio transcriptions for 1 hour
    
    # Pydantic v2 config for loading the .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    #model:
    MODEL_NAME : str = "whisper-tiny"
    
    #local cache config:
    LOCAL_CACHE_SIZE: int = 1024
    LOCAL_CACHE_MAX_SIZE : int = 1024
    LOCAL_CACHE_TTL_SECONDS : int = 3600 
    
    #device config:
    DEVICE : str = "mps" #macOS gpu hardware acceleration
    COMPUTE_TYPE : str = "float32"
    
    #sampling rate:
    SAMPLING_RATE : int = 16000
    
settings = Settings()
