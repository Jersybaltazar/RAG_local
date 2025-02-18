from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.mongodb import mongodb
from app.db.postgres import engine 
from app.ml.predictor import SalesPredictor
from app.models.postgres_reports import Base
from app.routers import predictions , sales , reports , chats

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar conexiones
    await mongodb.connect()
    # 2. Crear tablas en PostgreSQL
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    predictor = SalesPredictor()     
    await predictor.train_model()
       
    yield
    # Cerrar conexiones
    await mongodb.close()

app = FastAPI(lifespan=lifespan)

# Registrar routers
app.include_router(sales.router)
app.include_router(reports.router)
app.include_router(predictions.router)
app.include_router(chats.router)

@app.get("/")
async def root():
    return {"message": "Sistema de Ventas Inteligente en Linea"}