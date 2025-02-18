from datetime import datetime
from app.models.mongo_sales import SalesDocument
from app.models.postgres_reports import MonthlySalesReport
from app.db.postgres import get_async_session
from sqlalchemy.dialects.postgresql import insert

class SalesAggregator:
    async def aggregate_sales(self):
        try:
            # Obtener datos brutos de MongoDB
            sales = await SalesDocument.find_all().to_list()
            
            # Procesar y agrupar datos
            aggregated_data = self._process_sales(sales)
            
            # Guardar en PostgreSQL
            await self._save_to_postgres(aggregated_data)
            
            return {"status": "success", "processed_records": len(sales)}
            
        except Exception as e:
            raise RuntimeError(f"Aggregation error: {str(e)}")

    def _process_sales(self, sales_data):
        aggregated = {}
        
        for sale in sales_data:
            # Convertir fecha a objeto datetime (si no lo está ya)
            sale_date = sale.fecha if isinstance(sale.fecha, datetime) else datetime.fromisoformat(sale.fecha)
            
            key = (sale_date.strftime("%Y-%m"), sale.producto)
            
            if key not in aggregated:
                aggregated[key] = {
                    "total_sales": 0.0,
                    "units_sold": 0
                }
                
            aggregated[key]["total_sales"] += sale.total
            aggregated[key]["units_sold"] += sale.cantidad
            
        return [
            {
                "month": datetime.strptime(f"{key[0]}-01", "%Y-%m-%d").date(),  # Convertir a date
                "product": key[1],
                **values
            } for key, values in aggregated.items()
        ]
    async def _save_to_postgres(self, data):
        async for session in get_async_session():
            try:
                # Construir lista de valores
                values_list = [{
                    "month": record["month"],
                    "product": record["product"],
                    "total_sales": record["total_sales"],
                    "units_sold": record["units_sold"]
                } for record in data]

                # Crear statement batch
                stmt = insert(MonthlySalesReport).values(values_list)

                # Cláusula de actualización
                stmt = stmt.on_conflict_do_update(
                    constraint='idx_month_product',
                    set_={
                        'total_sales': stmt.excluded.total_sales,
                        'units_sold': stmt.excluded.units_sold
                    }
                )

                await session.execute(stmt)
                await session.commit()
                break

            except Exception as e:
                await session.rollback()
                raise e