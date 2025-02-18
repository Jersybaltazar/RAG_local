from fastapi import APIRouter, HTTPException, status
from app.ml.predictor import SalesPredictor
from typing import Optional
from app.models.mongo_sales import SalesDocument

router = APIRouter()
predictor = SalesPredictor()

@router.get("/predict-sales", 
           response_model=dict,
           description="Predice ventas futuras usando modelo ARIMA")
async def predict_sales(
    months: Optional[int] = 1,
    retrain: Optional[bool] = False
):
    try:
        sales_count = await SalesDocument.count()
        if sales_count < 30:
            raise HTTPException(
                status_code=422,
                detail="Se requieren mínimo 30 registros para predicciones"
            )
            
        await predictor.train_model(force_retrain=retrain)
        predictions = await predictor.predict(periods=30 * months)  # 30 días por mes
        
        return {
            "model": "ARIMA(5,1,0)",
            "horizon": f"{months} meses",
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error de predicción: {str(e)}"
        )