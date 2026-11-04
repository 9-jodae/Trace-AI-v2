import argparse
import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import AiUnavailableError
from app.ingestion.schemas import DocumentSubmission
from app.ingestion.service import prepare_document, store_document
from app.retrieval.neo4j_store import DocumentVersionConflict
from app.retrieval.service import close_stores


async def run(path: Path, normalize_only: bool, output: Path | None) -> None:
    document = DocumentSubmission.model_validate_json(path.read_text(encoding="utf-8"))
    normalized = await prepare_document(document)
    try:
        result = None if normalize_only else await store_document(normalized)
        if output is not None:
            # 명시적으로 요청한 원문 포함 저장 파일이다. stdout/로그로 원문을 출력하지 않는다.
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(normalized.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        summary = result.model_dump(mode="json", by_alias=True) if result else {
            "documentId": document.document_id,
            "classificationStatus": normalized.classification_status,
            "reviewReasons": normalized.review_reasons,
            "stored": False,
        }
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        await close_stores()


def main() -> None:
    parser = argparse.ArgumentParser(description="추출된 원문 JSON을 분류·검증해 Neo4j/Qdrant에 저장한다")
    parser.add_argument("path", type=Path)
    parser.add_argument("--normalize-only", action="store_true", help="DB 저장 없이 정규화만 수행")
    parser.add_argument("--output", type=Path, help="정규화 결과 저장 위치. 문서 원문이 포함되므로 커밋하지 말 것")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.path, args.normalize_only, args.output))
    except (ValidationError, ValueError, OSError, AiUnavailableError, DocumentVersionConflict):
        # 예외 문자열에는 입력 원문·파일 위치·외부 오류가 포함될 수 있다.
        parser.exit(1, "문서 처리 실패: 입력 규격·문서 버전·저장소 설정을 확인하세요.\n")


if __name__ == "__main__":
    main()
