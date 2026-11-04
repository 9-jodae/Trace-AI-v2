# 개발 규약

- 공통 코드는 `app/core`, 기능 코드는 `app/<기능>`에 둡니다.
- `router.py`는 HTTP 매핑과 검증, `schemas.py`는 요청·응답 모델, 나머지 모듈은 기능 로직을 맡습니다.
- JSON 필드는 `CamelModel`로 camelCase를 사용합니다. 오류 응답은 `{ "code": "...", "message": "..." }`이며 내부 예외 텍스트를 노출하지 않습니다.
- 스키마 변경은 `docs199-Server` 저장소의 Flyway 마이그레이션으로만 수행합니다. AI 서버는 스키마를 변경하지 않습니다.
- 결정 로직은 단위 테스트로 확인하고 JEV·LLM·임베딩·DB 호출은 테스트 대역으로 대체합니다.
- API 키, 자격 증명, 문서 본문, 비공개 파일 URL은 로그에 남기지 않습니다.

**결정 필요:** Spring의 `{ success, data, error }` 응답 형식을 서버 간 API에도 적용할지는 AI 연동 계약에서 정합니다.
