"""
Módulo de ingesta de tickets de supermercado (Argentina).
Procesa imágenes de tickets y devuelve datos estructurados listos para SQL.
"""

import base64
import logging
import os
import sys
from pathlib import Path

import dotenv
from openai import OpenAI
from pydantic import ValidationError

from models import ReceiptData

# Extensiones soportadas para imágenes
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Margen de error permitido para validación de totales (1%)
TOTAL_TOLERANCE = 0.01

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_api_key() -> str:
    """Carga OPENAI_API_KEY desde .env."""
    dotenv.load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. Crea un archivo .env con OPENAI_API_KEY=tu_clave"
        )
    return key


def encode_image_to_base64(image_path: str | Path) -> str:
    """
    Codifica una imagen en base64 para envío a la API.
    Soporta .jpg, .jpeg, .png y .webp.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato no soportado: {suffix}. "
            f"Soportados: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    with open(path, "rb") as f:
        data = f.read()

    return base64.standard_b64encode(data).decode("utf-8")


def get_media_type(path: str | Path) -> str:
    """Devuelve el media type para la API según la extensión."""
    suffix = Path(path).suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mapping.get(suffix, "image/jpeg")


SYSTEM_PROMPT = """Eres un experto en retail argentino. Analiza la imagen del ticket de supermercado y extrae todos los datos de forma estructurada.

Reglas:
- Identifica promociones y descuentos.
- Si un producto tiene un descuento, el unit_price debe reflejar el precio final pagado por unidad (no el precio tachado).
- El name debe ser normalizado y legible (ej: "Leche La Serenísima 1L" en lugar de "LECH LS 1L").
- date debe estar en formato ISO (YYYY-MM-DD).
- total_line_price = quantity * unit_price para cada ítem.
- Incluye cuit solo si aparece en el ticket."""


def parse_receipt_image(image_path: str | Path, client: OpenAI | None = None) -> ReceiptData:
    """
    Procesa una imagen de ticket y devuelve ReceiptData estructurado.
    Usa OpenAI Structured Outputs (parse) con el modelo gpt-5.2-mini.
    """
    image_path = Path(image_path)
    b64 = encode_image_to_base64(image_path)
    media_type = get_media_type(image_path)

    api_client = client or OpenAI(api_key=load_api_key())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    },
                },
            ],
        },
    ]

    # Structured Outputs: beta en SDK 2026, fallback a chat.completions.parse
    try:
        completion = api_client.beta.chat.completions.parse(
            model="gpt-5.2-mini",
            messages=messages,
            response_format=ReceiptData,
        )
    except AttributeError:
        completion = api_client.chat.completions.parse(
            model="gpt-5.2-mini",
            messages=messages,
            response_format=ReceiptData,
        )

    parsed: ReceiptData = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("La API no devolvió datos estructurados (parsed es None)")

    return parsed


def validate_totals(receipt: ReceiptData) -> None:
    """
    Valida que la suma de total_line_price coincida con total_amount
    dentro de un margen del 1%. Imprime warning si no coincide.
    """
    items_total = sum(item.total_line_price for item in receipt.items)
    diff = abs(receipt.total_amount - items_total)
    if receipt.total_amount == 0:
        if items_total != 0:
            logger.warning(
                "Validación: total_amount es 0 pero suma de ítems = %.2f",
                items_total,
            )
        return

    relative_error = diff / receipt.total_amount
    if relative_error > TOTAL_TOLERANCE:
        logger.warning(
            "Validación: suma de ítems (%.2f %s) no coincide con total_amount (%.2f %s). "
            "Diferencia: %.2f (%.2f%%)",
            items_total,
            receipt.currency,
            receipt.total_amount,
            receipt.currency,
            diff,
            relative_error * 100,
        )


def run_ingestion(image_path: str | Path) -> ReceiptData | None:
    """
    Punto de entrada: procesa la imagen y devuelve ReceiptData o None si hay error.
    Captura errores de conexión, validación Pydantic y lectura de imagen.
    """
    try:
        logger.info("Procesando imagen: %s", image_path)
        receipt = parse_receipt_image(image_path)
        validate_totals(receipt)
        logger.info(
            "OK - Comercio: %s, Fecha: %s, Total: %.2f %s, Ítems: %d",
            receipt.store_name,
            receipt.date,
            receipt.total_amount,
            receipt.currency,
            len(receipt.items),
        )
        return receipt
    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", e)
        return None
    except ValueError as e:
        logger.error("Error de formato o validación: %s", e)
        return None
    except ValidationError as e:
        logger.error("Error de validación Pydantic: %s", e)
        return None
    except Exception as e:
        logger.exception("Error de conexión o inesperado: %s", e)
        return None


def main() -> None:
    """CLI: recibe ruta de imagen como argumento."""
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_imagen_ticket>")
        print("Formatos soportados: .jpg, .png, .webp")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_ingestion(image_path)
    if result is None:
        sys.exit(1)

    # Salida estructurada (opcional: para piping o integración)
    print("\n--- ReceiptData (listo para SQL) ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
