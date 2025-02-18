import asyncio
import numpy as np
from datetime import datetime, timedelta
from app.models.mongo_sales import SalesDocument
from app.db.mongodb import mongodb

async def main():
    await mongodb.connect()
    
    # Configurar parámetros
    start_date = datetime(2024, 1, 1)
    days = 60  # 2 meses
    base_price = 800
    
    # Generar datos con tendencia + estacionalidad
    for i in range(days):
        # Variación semanal (más ventas los fines de semana)
        weekday = (start_date + timedelta(days=i)).weekday()
        weekend_factor = 1.5 if weekday >= 5 else 1.0
        
        # Tendencia lineal + ruido aleatorio
        quantity = int(10 + (i * 0.2) + (5 * weekend_factor) + np.random.normal(0, 2))
        
        sale = SalesDocument(
            fecha=start_date + timedelta(days=i),
            producto="Laptop",
            cantidad=max(5, quantity),  # Mínimo 5 unidades
            precio_unitario=base_price,
            total=max(5, quantity) * base_price
        )
        await sale.insert()
    
    print(f"✅ Insertados {days} registros sintéticos!")
    await mongodb.close()

if __name__ == "__main__":
    asyncio.run(main())