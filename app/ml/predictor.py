import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from app.models.mongo_sales import SalesDocument
from app.core.config import settings
from statsmodels.tsa.stattools import adfuller
class SalesPredictor:
    def __init__(self):
        self.model = None
        self.last_training_date = None
        self.history = None

    async def _load_data(self):
        """Carga y preprocesa datos históricos desde MongoDB"""
        sales = await SalesDocument.find_all().to_list()

        if len(sales) < 30:  # Mínimo 30 días de datos
            raise ValueError("Insuficientes datos para entrenar el modelo")
        
        df = pd.DataFrame([{
            'date': sale.fecha,
            'total': sale.total
        } for sale in sales])
        
        ts = df.set_index('date')['total'].resample('D').sum()
        ts = ts.replace(0, np.nan).ffill().fillna(0)
        if ts.var() < 1e-5:  # Evitar series constantes
            ts.iloc[-1] += 0.001  # Pequeña perturbación
        return ts
    
    async def train_model(self, force_retrain=False):
        try:
            if not force_retrain and self.model:
                days_since_last_train = (pd.Timestamp.now() - self.last_training_date).days
                if days_since_last_train < 7:
                    return
            
            ts = await self._load_data()
            
            # Verificar estacionariedad
            result = adfuller(ts)
            if result[1] > 0.05:  # Si no es estacionaria
                ts = ts.diff().dropna()
                
            # Auto-selección de parámetros segura
            self.model = ARIMA(ts, order=(1, 0, 0))  # Empezar con AR(1)
            self.model_fit = self.model.fit(method='innovations_mle')  # Método más estable
            
            self.last_training_date = pd.Timestamp.now()
            self.history = ts
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    async def predict(self, periods: int = 6):
        """Genera predicciones futuras"""
        if not self.model:
            await self.train_model()
            
        forecast = self.model_fit.forecast(steps=periods)
        return self._format_predictions(forecast)

    def _format_predictions(self, forecast):
        """Formatea las predicciones para la respuesta API"""
        history_dates = pd.date_range(
            start=self.history.index[-1],
            periods=len(self.history) + 1,
            freq='D'
        )[1:]
        
        future_dates = pd.date_range(
            start=history_dates[-1],
            periods=len(forecast) + 1,
            freq='D'
        )[1:]
        
        return {
            "historical": [{"date": d.strftime("%Y-%m-%d"), "total": v} 
                         for d, v in zip(history_dates, self.history)],
            "forecast": [{"date": d.strftime("%Y-%m-%d"), "total": v} 
                        for d, v in zip(future_dates, forecast)]
        }