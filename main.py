"""
Módulo de ingesta de tickets de supermercado (Argentina).
Procesa imágenes y PDFs de tickets y devuelve datos estructurados listos para SQL.
"""

import base64
from database import init_db, save_receipt_to_db
import logging
import os
import sys
from pathlib import Path

import dotenv
import fitz
from openai import OpenAI
from pydantic import ValidationError

from models import ReceiptData
from database import init_db, save_receipt_to_db

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


def load_file_content(file_path: str | Path) -> tuple[str, str]:
    """Devuelve (contenido, tipo) donde contenido es base64 o texto plano."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # EXTRACCIÓN DIRECTA DE TEXTO (Mucho más preciso para tickets digitales)
        doc = fitz.open(path)
        text = chr(12).join([page.get_text() for page in doc])
        doc.close()
        return text, "text/plain"
    
    # Si es imagen, sigue con base64
    return encode_image_to_base64(path), get_media_type(path)

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
Convertir los datos de un ticket de consumo en un objeto JSON estructurado con precisión contable.

### EXTRACTION & NORMALIZATION RULES:
1.  **PRODUCT_NORMALIZATION (CRITICAL):**
    - No copies el texto críptico (ej: "LECH.LS.DS.1L"). 
    - Normaliza a nombres legibles (ej: "Leche La Serenísima Descremada 1L").
    - Identifica y separa la Marca y la Presentación.

2.  **FINANCIAL_INTEGRITY:**
    - `unit_price`: Debe ser el precio NETO pagado por unidad después de descuentos. 
    - `total_line_price`: Debe ser estrictamente `quantity * unit_price`.
    - Ignora ítems con valor 0 o leyendas informativas.

3.  **DISCOUNT_LOGIC:**
    - Si hay un descuento general al final (ej: "10% dto-MAYOR 60"), aplícalo proporcionalmente al `unit_price` de cada producto para que la suma de `total_line_price` coincida con el `total_amount`.

4.  **METADATA (store_name MUY IMPORTANTE):**
    - `date`: Formato ISO (YYYY-MM-DD).
    - `store_name`: Debe ser el **nombre de la cadena/comercio**, no la sucursal ni el barrio.
        - Si aparece algo como "Carrefour Olivos", usar `"Carrefour"`.
        - Prioriza palabras junto al logo o en la cabecera como marca principal.
        - Nunca uses solo el nombre de la localidad (ej: "Olivos", "San Isidro") como `store_name`.
    - `cuit`: Solo números.

### BEHAVIORAL CONSTRAINTS:
- Si el input es texto (de un PDF), procésalo con rigor matemático.
- Si un campo es ilegible, usa "". No inventes datos.
- Formato de salida: JSON estricto según ReceiptData.
"""


def parse_receipt_image(file_path: str | Path, client: OpenAI | None = None) -> ReceiptData:
    content, media_type = load_file_content(file_path) # Usar la nueva función
    
    # Construir el mensaje según el tipo de contenido
    user_content = []
    if media_type == "text/plain":
        user_content.append({"type": "text", "text": f"Analiza este texto de ticket:\n\n{content}"})
    else:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{content}", "detail": "high"}
        })

    # IMPORTANTE: Para estos tickets largos, usa el modelo "grande" (gpt-4o por defecto)
    model = os.getenv("OPENAI_RECEIPT_MODEL", "gpt-4o")

    # Cliente de OpenAI (permite inyectar uno externo en tests)
    api_client = client or OpenAI(api_key=load_api_key())

    completion = api_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=ReceiptData,
    )
    parsed: ReceiptData = completion.choices[0].message.parsed

    # Post-procesar store_name para preferir marca conocida sobre razón social / sucursal
    # Usamos tanto el texto completo (si lo hay) como el store_name que devolvió el modelo.
    try:
        full_text = content if media_type == "text/plain" else ""
        combined = (full_text + " " + (parsed.store_name or "")).lower()
        known_brands = ["carrefour", "carrefour market", "jumbo", "disco", "coto"]
        for brand in known_brands:
            if brand in combined:
                parsed = parsed.model_copy(update={"store_name": brand.title()})
                break
    except Exception:
        # Si algo falla en el post-procesado, devolvemos lo que ya teníamos
        pass

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

    # 1) Inicializar BD (idempotente)
    init_db()

    # 2) Procesar ticket
    result = run_ingestion(file_path)
    if result is None:
        sys.exit(1)

    # 3) Guardar en BD
    save_receipt_to_db(result)
    print("✅ Datos guardados en superfollow.db")

    # 4) Salida estructurada (opcional: para piping o integración)
    print("\n--- ReceiptData (listo para SQL) ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
