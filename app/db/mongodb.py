from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from beanie import init_beanie
from app.models.mongo_sales import SalesDocument

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        if ":27017" in settings.MONGODB_URL:
            fixed_url = settings.MONGODB_URL.replace(":27017", "")           
            self.client = AsyncIOMotorClient(str(fixed_url))
        else:
            self.client = AsyncIOMotorClient(str(settings.MONGODB_URL))

        self.db = self.client["llm_db"] 

        await init_beanie(
            database=self.db,
            document_models=[SalesDocument]  # Aquí puedes añadir más modelos
        )

        print("Connectado a MongoDB!")


    async def close(self):
        if self.client:
            self.client.close()
        print("Desconectado de MongoDB")

mongodb = MongoDB()