from fastapi import APIRouter, HTTPException, status, Depends, FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any,AsyncIterator
from app.langchain.sales_chain import SalesChatEngine
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# Instancia única inicializada al arrancar la app
chat_engine = SalesChatEngine()

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Cuántas laptops se vendieron en marzo y predice las ventas para abril?",
                "chat_history": [
                    {"role": "user", "content": "¿Producto más vendido en 2023?"},
                    {"role": "assistant", "content": "El producto más vendido fue..."}
                ]
            }
        }

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, str]]  # Fuentes históricas
    prediction_data: Optional[Dict[str, Any]] = None  # Datos crudos de predicción
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Ejemplo de pregunta",
                "answer": "Respuesta generada",
                "sources": [{"source": "venta", "fecha": "2024-03-01"}],
                "prediction_data": {"periodos": 3, "proyeccion": [15000, 16000, 17000]}
            }
        }

@asynccontextmanager
async def chat_lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Inicializar el motor al iniciar
    await chat_engine.initialize()
    yield
    # Liberar recursos al cerrar (ej: conexiones a DB)
    await chat_engine.close()

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    lifespan=chat_lifespan  # <--- ¡Aquí se integra!
)

@router.post(
    "",
    response_model=ChatResponse,
    responses={
        500: {"description": "Error interno del sistema"},
        400: {"description": "Petición inválida"}
    }
)
async def chat_with_sales_data(request: ChatRequest):
    """
    Endpoint para consultar datos históricos y predicciones de ventas usando lenguaje natural.
    
    - **question**: Pregunta en texto libre (ej: "Ventas en marzo 2024")
    - **chat_history**: Historial de conversación para contexto conversacional
    """
    try:
        # Convertir historial al formato requerido por LangChain
        langchain_history = [
            (msg.content, next_msg.content) 
            for msg, next_msg in zip(
                [m for m in request.chat_history if m.role == "user"],
                [m for m in request.chat_history if m.role == "assistant"]
            )
        ] if request.chat_history else []

        # Procesar consulta
        response = await chat_engine.ask(
            question=request.question,
            chat_history=langchain_history
        )
        
        # Construir respuesta estructurada
        return {
            "question": request.question,
            "answer": response.get("answer", "No se pudo generar respuesta"),
            "sources": response.get("sources", []),
            "prediction_data": response.get("prediction_data")
        }

    except ValueError as e:
        logger.warning(f"Error de validación: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error interno: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error procesando la solicitud. Por favor intente nuevamente."
        )