"""
Módulo de ingesta de tickets de supermercado (Argentina).
Procesa imágenes y PDFs de tickets y devuelve datos estructurados listos para SQL.
"""

import base64
import logging
import os
import sys
from pathlib import Path

import dotenv
import fitz
from openai import OpenAI
from pydantic import ValidationError

from models import ReceiptData

# Cargar .env al inicio para que SAVE_RENDERED_IMAGE y OPENAI_API_KEY estén disponibles
dotenv.load_dotenv()

# Extensiones soportadas para imágenes
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Extensiones soportadas para PDF
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
# Todas las extensiones aceptadas
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

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
    """Carga OPENAI_API_KEY desde .env (ya cargado al importar el módulo)."""
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
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Formato de imagen no soportado: {suffix}. "
            f"Soportados: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    with open(path, "rb") as f:
        data = f.read()

    return base64.standard_b64encode(data).decode("utf-8")


def pdf_first_page_to_base64(pdf_path: str | Path, save_rendered_to: str | Path | None = None) -> str:
    """
    Convierte la primera página de un PDF a imagen PNG y la devuelve en base64.
    Usa 400 DPI para buena legibilidad del texto. Si save_rendered_to está definido, guarda ahí el PNG.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if path.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
        raise ValueError(f"Formato no soportado: {path.suffix}. Soportado: .pdf")

    doc = fitz.open(path)
    try:
        if len(doc) == 0:
            raise ValueError("El PDF no tiene páginas")

        page = doc[0]
        logger.info("PDF: %d página(s) total, procesando primera página", len(doc))

        # Recortar al área con contenido para quitar blanco y que el modelo vea solo el ticket
        try:
            words = page.get_text("words")
            if words:
                x0 = min(w[0] for w in words)
                y0 = min(w[1] for w in words)
                x1 = max(w[2] for w in words)
                y1 = max(w[3] for w in words)
                clip_rect = fitz.Rect(x0, y0, x1, y1)
                clip_rect = clip_rect + (-10, -20, 10, 40)
                clip_rect &= page.rect
            else:
                clip_rect = page.rect
        except Exception:
            clip_rect = page.rect

        # Limitar tamaño para que la API no redimensione y pierda detalle (máx ~4096 px por lado)
        max_side_px = 4096
        zoom = 300 / 72
        w_pt, h_pt = clip_rect.width, clip_rect.height
        if w_pt * zoom > max_side_px or h_pt * zoom > max_side_px:
            zoom = min(max_side_px / w_pt, max_side_px / h_pt, zoom)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

        png_bytes = pix.tobytes(output="png")
        size_mb = len(png_bytes) / (1024 * 1024)
        logger.info("PDF renderizado: imagen PNG de %.2f MB (%dx%d px)", size_mb, pix.width, pix.height)

        if save_rendered_to:
            Path(save_rendered_to).write_bytes(png_bytes)
            logger.info("Imagen guardada para revisión: %s", save_rendered_to)

        return base64.standard_b64encode(png_bytes).decode("utf-8")
    finally:
        doc.close()


def load_file_to_base64(file_path: str | Path) -> tuple[str, str]:
    """
    Carga un archivo (imagen o PDF) y devuelve (base64, media_type).
    Para PDFs convierte la primera página a PNG de alta calidad.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato no soportado: {suffix}. "
            f"Soportados: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        logger.info("Procesando PDF: %s", path.name)
        save_to = None
        if os.getenv("SAVE_RENDERED_IMAGE"):
            # Guardar junto al PDF con ruta absoluta para que sepas dónde está
            save_to = path.resolve().with_name(path.stem + "_rendered.png")
        b64 = pdf_first_page_to_base64(path, save_rendered_to=save_to)
        return b64, "image/png"
    
    logger.info("Procesando imagen: %s", path.name)
    b64 = encode_image_to_base64(path)
    return b64, get_media_type(path)


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


SYSTEM_PROMPT = """
### ROLE: SENIOR_RETAIL_DATA_EXTRACTOR (ARGENTINA)

### MISSION:
Convertir una imagen de ticket de consumo (supermercado/retail) en un objeto JSON estructurado con precisión contable.

### OBLIGATORIO - items NO puede quedar vacío:
- En `items` debes incluir TODAS las líneas de productos que aparecen en el ticket.
- Cada fila que tenga descripción de producto + cantidad + precio (o total de línea) es un ítem.
- Si el ticket tiene productos impresos, la lista `items` debe tener al menos un ítem por cada producto. No devuelvas `items: []` si hay líneas de productos visibles.

### EXTRACTION & NORMALIZATION RULES:
1.  **PRODUCT_NORMALIZATION (CRITICAL):**
    - No copies el texto críptico del ticket (ej: "LECH.LS.DS.1L").
    - Expande y normaliza a nombres humanos legibles (ej: "Leche La Serenísima Descremada 1L").
    - Identifica y separa la Marca y la Presentación (litros, gramos, unidades).

2.  **FINANCIAL_INTEGRITY:**
    - `unit_price`: Precio NETO por unidad después de descuentos. Si hay precio base y descuento: (Precio_Base - Descuento) / Cantidad.
    - `total_line_price`: estrictamente `quantity * unit_price`.
    - Ignora ítems informativos con valor 0 o leyendas (ej: "Usted ahorró...").

3.  **DISCOUNT_LOGIC:**
    - Si hay descuento general al final (ej: "Promo Banco 10%"), distribuye proporcionalmente en `unit_price` para que la suma de ítems coincida con `total_amount`.

4.  **METADATA:**
    - `date`: Formato ISO (YYYY-MM-DD). Si hay fecha en el ticket, extraerla siempre. Si no hay, usar string vacío "" (nunca "null").
    - `store_name`: Nombre comercial conocido (ej: "Coto", "Carrefour").
    - `total_amount`: Debe ser el monto total que figura en el ticket. Si el ticket muestra un total, no devolver 0.
    - `cuit`: Solo números si aparece.

### BEHAVIORAL CONSTRAINTS:
- No inventar datos. Solo extraer lo que está escrito.
- Completitud: rellenar todos los campos del esquema para los que el ticket tenga información (items, total_amount, date, store_name).
- Formato de salida: JSON estricto según el esquema ReceiptData.
"""


def parse_receipt_image(file_path: str | Path, client: OpenAI | None = None) -> ReceiptData:
    """
    Procesa una imagen o PDF de ticket y devuelve ReceiptData estructurado.
    Para PDFs se usa la primera página. Usa OpenAI Structured Outputs (parse).
    """
    file_path = Path(file_path)
    b64, media_type = load_file_to_base64(file_path)

    api_client = client or OpenAI(api_key=load_api_key())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Aplica el rol SENIOR_RETAIL_DATA_EXTRACTOR: extrae store_name, date, total_amount, cuit si aplica, y en items cada línea de producto con name, quantity, unit_price y total_line_price. No dejes items vacío si hay productos; no dejes total_amount en 0 si el ticket muestra un total.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    # Modelo: gpt-4o-mini tiene mejor visión para tickets; se puede sobreescribir con OPENAI_RECEIPT_MODEL
    model = os.getenv("OPENAI_RECEIPT_MODEL", "gpt-4o-mini")

    # Structured Outputs: beta en SDK 2026, fallback a chat.completions.parse
    try:
        completion = api_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ReceiptData,
        )
    except AttributeError:
        completion = api_client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ReceiptData,
        )

    parsed: ReceiptData = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("La API no devolvió datos estructurados (parsed es None)")

    # Normalizar date: si el modelo devolvió la palabra "null", usar string vacío
    if parsed.date and parsed.date.strip().lower() == "null":
        parsed = parsed.model_copy(update={"date": ""})

    # Validar que no sean valores de error del modelo
    if parsed.store_name.startswith("ERROR_") or parsed.date.startswith("ERROR_"):
        error_msg = (
            f"El modelo reportó error al procesar la imagen/PDF. "
            f"Store: {parsed.store_name}, Date: {parsed.date}. "
            f"Posibles causas: imagen de baja calidad, PDF corrupto, o formato no reconocido."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

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


def run_ingestion(file_path: str | Path) -> ReceiptData | None:
    """
    Punto de entrada: procesa imagen o PDF de ticket y devuelve ReceiptData o None si hay error.
    Captura errores de conexión, validación Pydantic y lectura de archivo.
    """
    try:
        logger.info("Procesando archivo: %s", file_path)
        receipt = parse_receipt_image(file_path)
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
    """CLI: recibe ruta de imagen o PDF como argumento."""
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_ticket>")
        print("Formatos soportados: .jpg, .png, .webp, .pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    result = run_ingestion(file_path)
    if result is None:
        sys.exit(1)

    # Salida estructurada (opcional: para piping o integración)
    print("\n--- ReceiptData (listo para SQL) ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
