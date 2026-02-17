"""
Modelos Pydantic para datos estructurados de tickets de supermercado (Argentina).
Todos los valores monetarios son float para inserción directa en SQL.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TicketItem(BaseModel):
    """Línea de producto en un ticket de supermercado."""

    name: str = Field(..., description="Nombre del producto normalizado")
    quantity: float = Field(..., ge=0, description="Cantidad")
    unit_price: float = Field(..., ge=0, description="Precio unitario final (post descuento)")
    total_line_price: float = Field(..., ge=0, description="Total de la línea (quantity * unit_price)")


class ReceiptData(BaseModel):
    """Datos estructurados de un ticket de supermercado listos para BD."""

    store_name: str = Field(..., description="Nombre del comercio")
    cuit: Optional[str] = Field(None, description="CUIT del emisor si está en el ticket")
    date: str = Field(..., description="Fecha en formato ISO (YYYY-MM-DD)")
    items: list[TicketItem] = Field(..., description="Lista de productos")
    total_amount: float = Field(..., ge=0, description="Monto total del ticket")
    currency: str = Field(default="ARS", description="Moneda (ARS por defecto)")
