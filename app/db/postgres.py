from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.core.config import settings

engine = create_async_engine(
    settings.POSTGRES_URL.replace('postgresql://', 'postgresql+asyncpg://'),
    connect_args={
        "ssl": "require"  # Requerido por Neon
    },
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,  # timeout en segundos
    pool_pre_ping=True,  # verifica que la conexión está viva
    echo=True # para ver las consultas SQL (útil para debug)
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session