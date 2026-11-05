from io import BytesIO

from pypdf import PdfReader


class UnsupportedDocumentError(Exception):
    pass


def extract_text(extension: str, data: bytes) -> str:
    # charStart, charEnd가 이 본문 기준이므로 docs199-Server의 export-documents-json.py와 규칙이 같아야 한다
    extension = extension.lower()
    if extension in {"md", "txt"}:
        return data.decode("utf-8")
    if extension == "pdf":
        return "\n\n".join((page.extract_text() or "").strip() for page in PdfReader(BytesIO(data)).pages)
    raise UnsupportedDocumentError(extension)
