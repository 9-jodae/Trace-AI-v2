# docs199-AI

근거를 제시하는 사내 문서 검색 도우미의 AI 기획과 개발을 관리하는 레포입니다.

Python AI 서버는 FastAPI와 LangGraph로 검색과 답변 생성을 처리합니다.

## 현재 구현과 기획 문서

이슈 #27 브랜치는 추출된 원문 JSON의 분류·검증·저장과 문서 중심 그래프 검색을 구현합니다. 원본 파일 파서·OCR, Notion/Drive 수집, Spring의 파일 적재 요청·콜백 연결은 아직 없습니다.

| 문서 | 내용 |
|---|---|
| [프로젝트 기획서](docs/프로젝트_기획서.md) | 구현·미구현 범위, 전체 구조와 남은 결정 |
| [문서 적재 설계](docs/문서_적재_설계.md) | 필드 추출, LLM 출력·원문 근거 검증, 검토 대기, API·CLI |
| [검색·저장소 설계](docs/검색_저장소_설계.md) | 문서 노드·참조 관계, Qdrant 근거 회수, RRF·권한·버전 필터 |
| [구현·검증 현황](docs/구현_검증_현황.md) | 실제 확인 결과와 실행하지 않은 검증 |
| [PostgreSQL 검토 초안](docs/postgresql_연동_설계.md) | 현재 구현과 구분한 기존 대안 |

AI 내부 텍스트 적재 API는 `POST /api/v1/ingestion/documents`입니다. 제목·본문·문서/버전 ID·버전 번호·상태·허용 그룹이 필요합니다. LLM은 문서 유형·주제·명시적 참조 후보만 반환하고, 원문 인용과 대상 문서를 검증합니다. 분류 불확실성은 검토 대기로 저장하되, 유효한 원문 검색은 허용합니다.

Neo4j에는 Document 노드와 REFERENCES를 저장하고, Qdrant에는 원문 Chunk·임베딩·메타데이터를 저장합니다. 신규 검색은 벡터 검색과 그래프가 찾은 문서 안의 Chunk 검색을 RRF로 융합합니다. 현재 구현의 권한·캐시 재검증 한계는 기획서에 명시했습니다.

```bash
python -m app.ingestion data/ingestion_document.example.json --normalize-only --output output/normalized.json
python -m app.ingestion data/ingestion_document.example.json
```

첫 명령은 DB 저장 없이 정규화 결과를 만들고, 둘째 명령은 실제 DB 저장까지 수행합니다. LLM 설정에 따라 모델 호출 비용이 발생할 수 있습니다. 실제 적재에는 `VECTOR_STORE=qdrant`와 두 DB·임베딩 설정이 필요합니다. `output/`에는 원문이 포함될 수 있어 커밋하지 않습니다.

## 로컬 실행

자세한 내용은 [로컬 실행 안내](docs/local-setup.md)를 참고하세요.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python main.py
```

`http://localhost:8000/docs`에서 API를, `http://localhost:8000/playground`에서 테스트 화면을 확인합니다.

## 개발

[개발 규약](docs/conventions.md)을 확인하세요. PR 전에는 `ruff check . && pytest`를 실행합니다.

서버 간 계약은 `docs199-Server` 저장소의 `docs/ai-server-contract.md`에 기록합니다. 아직 확정되지 않은 항목이 있습니다.
