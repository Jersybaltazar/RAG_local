from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms.gpt4all import GPT4All
from langchain.docstore.document import Document
from sqlalchemy import select
from app.models.postgres_reports import MonthlySalesReport
from typing import List, Tuple, Optional
from langchain.prompts import PromptTemplate
from app.db.postgres import AsyncSessionLocal
from app.ml.predictor import SalesPredictor  # Asume que tienes esta clase
from pathlib import Path
import json
import re
import logging
import os

logging.basicConfig(
    level=logging.INFO,  # Muestra solo INFO, WARNING y ERROR (no DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Deshabilitar logs excesivos de librerías externas (MongoDB, SQLAlchemy)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)  # MongoDB
logging.getLogger("uvicorn").setLevel(logging.WARNING)  # FastAPI Server
logging.getLogger("langchain").setLevel(logging.WARNING)  # LangChain interno

logger = logging.getLogger(__name__)

class SalesChatEngine:
    def __init__(self,model_path: str = "./models/llama-2-7b-chat.ggmlv3.q4_0.bin"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[GPT4All] = None
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.predictor = SalesPredictor()  # Instancia del modelo ARIMA
        self._initialized = False
        self.model_path = os.path.abspath(model_path)
        self.qa_prompt = PromptTemplate(
            template="""Eres un experto en ventas. Responde SOLO con los datos proporcionados. 
            Si no hay información relevante, di 'No tengo datos para responder esto'.

            Contexto:
            {context}

            Pregunta: {question}
            Respuesta útil:""",
            input_variables=["context", "question"]
        )
    async def initialize(self):
        """Inicialización única con manejo de errores mejorado"""
        if self._initialized:
            return
            
        try:
            # 1. Cargar y combinar datos
            sales_data = await self._load_sales_data()
            reports_data = await self._load_reports_data()
            logger.debug(f"Ventas cargadas: {sales_data[:2]}...")  # muestra muestra de datos
            logger.debug(f"Reportes cargados: {reports_data[:2]}...")
            # 2. Crear documentos con formato mejorado
            docs = self._create_documents(sales_data + reports_data)
            logger.info(f"Creados {len(docs)} documentos para el vector store")
            # 3. Crear o cargar vector store
            self.vector_store = await self._get_vector_store(docs)
            test_results = self.vector_store.similarity_search("prueba de venta", k=1)
            logger.info(f"Resultado de búsqueda de prueba en vector store: {test_results}")
            # 4. Inicializar LLM con validación
            self._init_llm()
            
            # 5. Configurar cadena de conversación
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": 0.3}
                ),
                combine_docs_chain_kwargs={'prompt': self.qa_prompt},
                return_source_documents=True,
                verbose=True
            )
            
            self._initialized = True
            logger.info("LangChain pipeline inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {str(e)}")
            raise

    async def _get_vector_store(self, docs: List[Document]) -> FAISS:
        """Obtiene o crea el vector store con caché en disco"""
        try:
            # Cambiar la normalización de los vectores
            vector_store = await FAISS.afrom_documents(
                docs, 
                self.embeddings,
                normalize_L2=True  # Añadir esta línea
            )
            return vector_store
        except Exception as e:
            logger.error(f"Error creando vector store: {str(e)}")
            raise

    def _init_llm(self):
        """Configuración validada del LLM"""
        try:
            self.llm = GPT4All(
                model=self.model_path,
                max_tokens=2048,
                n_threads=6,        # reduce el contexto de 2048 a 1024
                n_batch=128, 
                backend='llama', # Verificar compatibilidad con el modelo
            )
            # Test mínimo de funcionamiento
            test_response = self.llm.generate(["Hola"])
            if not test_response.generations:
                raise ValueError("El LLM no respondió correctamente")
                
        except Exception as e:
            logger.error(f"Error configurando LLM: {str(e)}")
            raise

    async def _load_sales_data(self) -> List[dict]:
        """Carga optimizada de ventas desde MongoDB"""
        from app.models.mongo_sales import SalesDocument
        
        try:
            sales = await SalesDocument.find_all().to_list()
            logger.info(f"Loaded {len(sales)} sales documents")
            return [{
                "type": "venta",
                "fecha": sale.fecha.isoformat(),
                "producto": sale.producto,
                "detalle": f"Venta de {sale.cantidad} {sale.producto} por ${sale.total} el {sale.fecha.date()}"
            } for sale in sales]
            
        except Exception as e:
            logger.error(f"Error cargando ventas: {str(e)}")
            return []

    async def _load_reports_data(self) -> List[dict]:
        """Carga segura de reportes con contexto asíncrono"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(MonthlySalesReport).order_by(MonthlySalesReport.month))
                reports = result.scalars().all()
                
                return [{
                    "type": "reporte",
                    "mes": report.month.strftime("%Y-%m"),
                    "producto": report.product,
                    "detalle": (
                        f"Reporte {report.month.strftime('%B %Y')}: "
                        f"{report.units_sold} unidades vendidas, "
                        f"Total: ${report.total_sales}"
                    )
                } for report in reports]  
            except Exception as e:
                logger.error(f"Error cargando reportes: {str(e)}")
                return []

    def _create_documents(self, data: List[dict]) -> List[Document]:
        """Crea documentos con contexto estructurado"""
        documents = []
        for item in data:
            content = (
                f"[Tipo: {item['type'].upper()}]\n"
                f"Fecha: {item.get('fecha', item.get('mes', 'N/A'))}\n"
                f"Producto: {item.get('producto', 'General')}\n"
                f"Detalles: {item['detalle']}\n"
                f"Fuente: {'Venta individual' if item['type'] == 'venta' else 'Reporte agregado'}"
            )
            metadata = {
                "source": item["type"],
                "producto": item.get("producto", ""),
                "fecha": item.get("fecha") or item.get("mes", "")
            }
            documents.append(Document(page_content=content, metadata=metadata))
        logger.debug(f"Documentos creados: {[doc.metadata for doc in documents]}")
        for i, doc in enumerate(documents[:5]):  # Solo mostramos los primeros 5
            logger.info(f"Documento {i+1}: {doc.page_content} | Metadata: {doc.metadata}")

# Ver los embeddings generados antes de guardarlos en FAISS
        try:
            sample_text = documents[0].page_content  # Tomamos un documento de prueba
            sample_embedding = self.embeddings.embed_query(sample_text)
            logger.info(f"Ejemplo de embedding generado: {sample_embedding[:5]}... (muestra de 5 valores)")
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")

        return documents
        # Ver los primeros documentos creados antes de indexarlos en FAISS

    async def ask(self, question: str, chat_history: List[Tuple[str, str]] = []) -> dict:
        """Manejo unificado de preguntas históricas y predictivas"""
        if not self._initialized:
            await self.initialize()
            
        # Detectar preguntas predictivas
        if self._is_prediction_question(question):
            return await self._handle_prediction(question)
            
        # Preguntas históricas
        return await self._handle_historical_question(question, chat_history)

    def _is_prediction_question(self, question: str) -> bool:
        """Determina si la pregunta requiere predicción"""
        prediction_keywords = [
            r'predi[ck]', r'proyect', r'estim', 
            r'futur', r'pr[óo]xim', r'ser[áa]'
        ]
        pattern = re.compile('|'.join(prediction_keywords), re.IGNORECASE)
        return bool(pattern.search(question))

    async def _handle_prediction(self, question: str) -> dict:
        """Ejecuta y formatea predicciones del modelo ARIMA"""
        try:
            # Extraer parámetros de la pregunta
            matches = re.findall(r'\d{4}-\d{2}', question)
            forecast_months = 6  # Default
            
            if matches:
                # Lógica para cálculo de periodos (implementar según necesidad)
                forecast_months = len(matches)
                
            predictions = await self.predictor.predict(forecast_months)
            
            return {
                "answer": (
                    f"Predicción para los próximos {forecast_months} meses:\n" +
                    "\n".join([f"- {m}: ${v:.2f}" for m, v in predictions])
                ),
                "sources": [{"type": "modelo", "name": "ARIMA"}]
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {"answer": "No pude generar la predicción", "sources": []}

    async def _handle_historical_question(self, question: str, chat_history) -> dict:
        """Procesa preguntas sobre datos históricos"""
        try:
             # Búsqueda manual en FAISS antes de la consulta real
            test_results = self.vector_store.similarity_search(question, k=5)
            logger.info(f"Resultados de FAISS para '{question}': {test_results}")
            
            response = await self.qa_chain.acall({
                "question": question,
                "chat_history": chat_history
            })
            
            # Formatear fuentes
            sources = []
            for doc in response["source_documents"]:
                source_type = "Venta" if doc.metadata["source"] == "venta" else "Reporte"
                sources.append({"source":source_type , "fecha":doc.metadata['fecha']})
                
            unique_sources = [] 
            for s in sources: 
                if s not in unique_sources: 
                    unique_sources.append(s)    
            return {
                "answer": response["answer"],
                "sources":unique_sources[:3] #  # Limitar a 3 únicos
            }
            
        except Exception as e:
            logger.error(f"Error en consulta histórica: {str(e)}")
            return {"answer": "Error obteniendo datos históricos", "sources": []}