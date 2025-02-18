from fastapi import APIRouter, HTTPException, status
from app.models.mongo_sales import SalesDocument
from app.models.schemas import SalesCreate, SalesResponse
from app.db.mongodb import mongodb

router = APIRouter(prefix="/sales", tags=["Ventas"])

@router.post("/", response_model=SalesResponse, status_code=status.HTTP_201_CREATED)
async def create_sale(sale: SalesCreate):
    try:
        # Calcular total automáticamente
        total = sale.cantidad * sale.precio_unitario
        sale_data = sale.model_dump()
        sale_data["total"] = total
        

        new_sale = SalesDocument(**sale_data)

        await new_sale.insert()
        
        return new_sale
    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating sale: {str(e)}"
        )