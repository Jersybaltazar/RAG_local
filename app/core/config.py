from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    MONGODB_URL: str = Field(
        default="mongodb+srv://jersybaltazarc:LjcmrEYxhmuoP2Gq@llm.4mbm1.mongodb.net/?retryWrites=true&w=majority&appName=llm",
        pattern=r"^mongodb\+srv://.*"
    ) 
    POSTGRES_URL: str = "postgresql+asyncpg://neondb_owner:npg_k9BX2tMyumZi@ep-lucky-term-a8xrt73y-pooler.eastus2.azure.neon.tech/neondb?ssl=require"
    
    LLM_MODEL_PATH: str = "./models/ggml-gpt4all-j-v1.3-groovy.bin"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()