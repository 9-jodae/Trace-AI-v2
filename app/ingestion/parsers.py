from io import BytesIO

from pypdf import PdfReader


class UnsupportedFormatError(ValueError):
    pass


def extract_text(extension: str, data: bytes) -> str:
    # 서버의 문서 내보내기(scripts/export-documents-json.py)와 같은 규칙이어야 근거 위치가 어긋나지 않는다
    kind = extension.lower()
    if kind in {"md", "txt"}:
        return data.decode("utf-8")
    if kind == "pdf":
        return "\n\n".join((page.extract_text() or "").strip() for page in PdfReader(BytesIO(data)).pages)
    raise UnsupportedFormatError(kind)
