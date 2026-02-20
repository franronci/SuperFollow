import sqlite3
from typing import Optional

from models import ReceiptData

DB_PATH = "superfollow.db"


def init_db() -> None:
    """Inicializa las tablas con integridad referencial e índices básicos."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Tabla de comercios
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS stores ("
            "id INTEGER PRIMARY KEY, "
            "name TEXT UNIQUE, "
            "cuit TEXT)"
        )
        # Tabla de productos (para normalización de precios)
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS products ("
            "id INTEGER PRIMARY KEY, "
            "name TEXT UNIQUE)"
        )
        # Tabla de tickets
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY,
                store_id INTEGER,
                date TEXT,
                total_amount REAL,
                currency TEXT,
                user_id INTEGER,
                FOREIGN KEY(store_id) REFERENCES stores(id)
            )
            """
        )
        # Detalle de productos por ticket
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS receipt_items (
                id INTEGER PRIMARY KEY,
                receipt_id INTEGER,
                product_id INTEGER,
                quantity REAL,
                unit_price REAL,
                total_line_price REAL,
                FOREIGN KEY(receipt_id) REFERENCES receipts(id),
                FOREIGN KEY(product_id) REFERENCES products(id)
            )
            """
        )

        # Índices para mejorar consultas típicas
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipts_store_date "
            "ON receipts(store_id, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipt_items_receipt_id "
            "ON receipt_items(receipt_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipt_items_product_id "
            "ON receipt_items(product_id)"
        )

        conn.commit()


def _get_single_id(
    cursor: sqlite3.Cursor,
    query: str,
    params: tuple,
    entity_name: str,
    value: str,
) -> int:
    """Devuelve el id de una fila, o lanza ValueError si no existe."""
    row = cursor.execute(query, params).fetchone()
    if row is None:
        raise ValueError(f"{entity_name} no encontrado en BD después de insertar: {value!r}")
    return int(row[0])


def save_receipt_to_db(data: ReceiptData, user_id: Optional[int] = None) -> None:
    """Guarda el objeto ReceiptData de forma relacional."""
    if not data.store_name or not data.store_name.strip():
        raise ValueError("ReceiptData.store_name está vacío; no se puede guardar el ticket.")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # 1. Store
        cursor.execute(
            "INSERT OR IGNORE INTO stores (name, cuit) VALUES (?, ?)",
            (data.store_name, data.cuit),
        )
        store_id = _get_single_id(
            cursor,
            "SELECT id FROM stores WHERE name=?",
            (data.store_name,),
            "Store",
            data.store_name,
        )

        # 2. Receipt
        cursor.execute(
            "INSERT INTO receipts (store_id, date, total_amount, currency, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (store_id, data.date, data.total_amount, data.currency, user_id),
        )
        receipt_id = cursor.lastrowid

        # 3. Items
        for item in data.items:
            if not item.name or not item.name.strip():
                raise ValueError("TicketItem.name está vacío; no se puede guardar el ítem.")

            cursor.execute(
                "INSERT OR IGNORE INTO products (name) VALUES (?)",
                (item.name,),
            )
            product_id = _get_single_id(
                cursor,
                "SELECT id FROM products WHERE name=?",
                (item.name,),
                "Product",
                item.name,
            )
            cursor.execute(
                """
                INSERT INTO receipt_items (
                    receipt_id,
                    product_id,
                    quantity,
                    unit_price,
                    total_line_price
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    product_id,
                    item.quantity,
                    item.unit_price,
                    item.total_line_price,
                ),
            )

        conn.commit()