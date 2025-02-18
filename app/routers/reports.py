from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, func, extract
from sqlalchemy.orm import aliased
from app.models.postgres_reports import MonthlySalesReport
from app.models.schemas import SalesReportResponse
from app.db.postgres import get_async_session
from app.services.agregator import SalesAggregator


router = APIRouter(prefix="/sales-report", tags=["Reportes"])

@router.get("/", response_model=list[SalesReportResponse])
async def get_sales_report():
    try:
        async for session in get_async_session():
            result = await session.execute(
                select(
                    MonthlySalesReport.month,
                    MonthlySalesReport.product,
                    MonthlySalesReport.total_sales,
                    MonthlySalesReport.units_sold
                )
            )
            reports = result.fetchall()
            
            return [{
                "month": str(report.month),
                "product": report.product,
                "total_sales": report.total_sales,
                "units_sold": report.units_sold
            } for report in reports]
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving reports: {str(e)}"
        )

@router.post("/aggregate", status_code=status.HTTP_202_ACCEPTED)
async def trigger_aggregation():
    try:
        aggregator = SalesAggregator()
        result = await aggregator.aggregate_sales()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )