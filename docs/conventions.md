# 개발 규약

- 공통 코드는 `app/core`, 기능 코드는 `app/<기능>`에 둡니다.
- `router.py`는 HTTP 매핑과 검증, `schemas.py`는 요청·응답 모델, 나머지 모듈은 기능 로직을 맡습니다.
- JSON 필드는 `CamelModel`로 camelCase를 사용합니다. 오류 응답은 `{ "code": "...", "message": "..." }`이며 내부 예외 텍스트를 노출하지 않습니다.
- `charStart`, `charEnd`는 적재 JSON `text`의 유니코드 코드 포인트 위치입니다(`docs199-Server`의 `docs/ai-server-contract.md`). BMP 밖 문자가 있는 문서는 Java UTF-16 위치와 다를 수 있어 별도 변환이 필요합니다.
- 스키마 변경은 `docs199-Server` 저장소의 Flyway 마이그레이션으로만 수행합니다. AI 서버는 스키마를 변경하지 않습니다.
- 결정 로직은 단위 테스트로 확인하고 JEV·LLM·임베딩·DB 호출은 테스트 대역으로 대체합니다.
- API 키, 자격 증명, 문서 본문, 비공개 파일 URL은 로그에 남기지 않습니다.

**결정 필요:** Spring의 `{ success, data, error }` 응답 형식을 서버 간 API에도 적용할지는 AI 연동 계약에서 정합니다.

## 서버 계약 확정을 기다리는 항목

추측해서 구현하지 않고, `docs199-Server#14`, `#15`에서 확정되면 반영합니다.

- 질문 요청으로 사용자 권한 범위(`AccessScope`)를 받는 JSON 형식
- 응답의 핵심 문구(`evidence`) 필드 이름과 Spring 전달 방식. 현재 형식은 임시입니다.
- PDF 페이지와 `bbox`(`chunks.page`, `chunks.location`). 파서 확정 후 저장합니다.
- `/ingest` 요청·콜백 필드와 원본 파일 전달 방식
- 대화 맥락 복원을 위한 `conversationId`와 이전 메시지 전달 방식
- 질문 길이 제한. 현재 2,000자는 임시값입니다.
- 답변 LLM의 Bedrock 모델

