import io

from pypdf import PdfReader
from pypdf.errors import PyPdfError


class ExtractionError(ValueError):
    """Raised when text cannot be extracted from an uploaded file."""


def extract_text(extension: str, data: bytes) -> str:
    if extension == ".txt":
        return data.decode("utf-8", errors="replace")
    if extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except PyPdfError as exc:
            raise ExtractionError(f"Could not read PDF: {exc}") from exc
    raise ExtractionError(f"Unsupported file type: {extension}")