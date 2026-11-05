# 로컬 실행

저장소를 복제한 뒤 `git config core.hooksPath .githooks`를 실행해 pre-push 훅을 활성화합니다. 훅은 원격 브랜치 삭제와 non-fast-forward push를 거부합니다. 이미 push한 브랜치는 rebase하지 말고 `git merge`로 갱신합니다.

1. Python 3.15로 가상 환경을 만들고 `pip install -r requirements-dev.txt`로 의존성을 설치합니다.
2. `.env.example`을 `.env`로 복사하고 값을 채웁니다. 각 항목의 설명은 `.env.example`에 있습니다.
3. `python main.py`를 실행합니다. 서버는 8000번 포트에서 시작합니다.

## 실행 환경

`APP_ENV=local`일 때만 개발 모드입니다. 값이 없거나 다른 값이면 운영 환경으로 처리합니다.

| 항목 | 로컬(`APP_ENV=local`) | 운영(그 외) |
| --- | --- | --- |
| `X-Internal-Secret` 검증 | 생략 | 필수. `INTERNAL_SECRET`이 없으면 서버가 시작되지 않음 |
| `/docs`, `/playground` | 열림 | 닫힘 |
| 샘플 데이터 | `USE_SAMPLE_DATA=true`일 때만 사용 | 사용 불가. 설정하면 서버가 시작되지 않음 |
| `python main.py` | `127.0.0.1`, 코드 변경 시 자동 재시작 | `0.0.0.0`, 자동 재시작 없음 |

샘플 데이터를 켜지 않으면 `DATABASE_URL`(pgvector)이 필요합니다. 없으면 검색 경로는 503을 반환하며, 샘플로 대체하지 않습니다. 그래프 검색은 선택이라 `NEO4J_URI`를 비우면 벡터 검색만 씁니다. `NEO4J_URI`를 넣으면 `NEO4J_PASSWORD`도 필요합니다. `ENTITY_EXTRACTION_ENABLED`는 기본 꺼짐입니다.

`DOCUMENT_TOPICS_ENABLED=true`이면 적재할 때 문서마다 앞부분 2,000자를 `LLM_MODEL`로 보내 주제를 1~3개 만들고 Neo4j에 저장합니다. 기본은 꺼짐입니다. 화면 간선으로 전달하는 방식은 `docs199-Server` 이슈에서 정합니다. `python -m app.retrieval.document_topics <JSON 파일>`은 저장 없이 주제와 간선을 출력합니다.

검색은 벡터와 그래프를 동시에 조회하고, 그래프 후보가 5개 미만이면 PostgreSQL 전문 검색을 더해 RRF로 합칩니다. 관계 그래프는 문서 제목(파일명 확장자를 제외한 표기도 포함)에 이름이 명시된 문서를 찾습니다. 본문 첫 줄이나 본문 문장은 문서 이름으로 복사하지 않습니다. 참조 표기는 `《문서명》` 또는 `[[문서명]]`이고, 이름이 중복되거나 대상이 없으면 관계를 만들지 않습니다. JSON 일괄 적재는 입력 문서와 기존 Neo4j 문서 제목 목록을 함께 사용하므로 일부 문서만 적재해도 이미 등록된 다른 파일의 대상을 찾습니다. 문서 전체와 권한은 Neo4j에 복사하지 않습니다. 다만 엔티티 추출을 켜면 근거 인용은 저장됩니다. 질문에서 찾은 시작 문서는 PostgreSQL `accessible_chunks`로 권한·현재 버전·문서 필터를 먼저 확인한 뒤 관계를 확장하고, 최종 청크 본문도 같은 함수로 다시 읽습니다. 그래프 후보는 관계 근거 청크를 우선하고 문서당 최대 5개로 제한합니다. 질문 임베딩이 있으면 그래프 후보도 기존 cosine distance 상한(기본 0.85)을 PostgreSQL 쿼리에서 적용합니다. 임베딩 장애 시에는 벡터 검색 실패를 표시하고 명시 관계 경로를 계속 사용할 수 있습니다. 전문 검색은 `simple` 설정이라 "연차는"처럼 조사가 붙은 단어는 "연차"와 맞지 않을 수 있습니다.

관계 검색은 아직 실제 Neo4j에서 검증하지 않았습니다. 팀 서버의 `NEO4J_URI`는 비워 둡니다. 검색 평가 문서의 같은 질문 세트로 그래프 적용 전후 순위를 비교하고, 개선이 확인된 뒤에만 서버에서 Neo4j를 켭니다.

벡터 검색은 pgvector cosine distance가 `VECTOR_MAX_COSINE_DISTANCE` 이하인 청크만 반환합니다. 기본값은 `0.85`입니다. 시연 문서 18개에 Titan v2로 질문 15개를 측정해 정했습니다. 관련 질문 9개의 정답 문서는 거리 0.435~0.821이었고, 무관 질문 6개의 1위는 0.827~0.950이었습니다. 표본이 작아 문서가 늘면 다시 재야 합니다. 맞는 문서를 놓치는 쪽이 더 나쁘다고 보아 여유를 두었고, 기준을 통과한 무관 질문은 답변 단계에서 거절합니다. 값은 0~2 범위여야 하고, 잘못된 값이면 서버가 시작되지 않습니다.

권한은 서버의 `accessible_chunks`와 `filter_accessible_chunk_ids`로 판단합니다. 그래프 검색 결과와 재사용할 이전 근거도 `filter_accessible_chunk_ids`를 거치므로, PostgreSQL이 멈추면 그래프 결과도 내보내지 않습니다.

질문 요청의 `filters`는 서버의 9인자 `accessible_chunks`로 벡터 검색과 그래프 결과에 적용합니다. 필터가 있는 요청은 이전 근거를 재사용하지 않고 새로 검색합니다. 샘플 데이터에는 필터가 적용되지 않습니다.

권한 범위는 질문 요청의 `accessScope`로 받습니다(`departmentId`, `position`, `clearance`, `isAdmin`). 요청에 없으면 운영 환경에서는 어떤 문서도 근거로 돌려주지 않습니다. 로컬 시험에서는 요청에 없을 때 `ACCESS_DEPARTMENT_ID`, `ACCESS_POSITION`, `ACCESS_CLEARANCE`, `ACCESS_IS_ADMIN`으로 권한 범위를 지정합니다. `ACCESS_CLEARANCE`나 `ACCESS_IS_ADMIN=true`가 없으면 권한 범위가 없는 것으로 봅니다.

JSON 문서는 `python -m app.retrieval.indexing <파일>`로 적재합니다. 현재 버전이 아닌 문서는 임베딩 전에 건너뛰고, 저장 직전에 다시 확인합니다. 파일 업로드와 파서는 아직 없습니다. `chunks.document_version_id`가 서버의 `document_versions.id`를 참조하므로 `version_id`는 서버에 있는 문서 버전의 UUID여야 합니다. `data/demo_documents.json`은 문자열 ID를 쓰고 있어 그대로는 팀 서버에 적재할 수 없습니다.

대화는 메모리에 저장하며 대화당 최근 메시지 6개와 컨텍스트만 보관합니다. 기본값은 최대 1,000개 대화, 1시간이고 `MAX_CONVERSATIONS`, `CONVERSATION_TTL_SECONDS`로 바꿀 수 있습니다. 서버를 재시작하면 사라집니다.

임베딩과 LLM을 Bedrock으로 쓰면 API 키 없이 AWS 자격 증명과 `BEDROCK_REGION`을 사용합니다. 비워 두면 AWS 기본 리전을 따릅니다. 임베딩은 질문과 문서 모두 Titan v2, 1024차원, `normalize=true`로 맞춥니다.

노트북에서는 `EMBEDDING_PROXY_URL`로 서버의 임시 임베딩 중계를 쓸 수 있습니다. `INTERNAL_SECRET` 값을 `X-Internal-Secret` 헤더로 보내고, 평문이 공개망을 지나지 않도록 SSH 터널(`-L 18880:localhost:18080`) 주소를 씁니다. 운영 환경에서는 이 설정을 무시합니다. 문서 일괄 적재는 팀 서버에서 직접 실행하는 것을 권장합니다.

## 운영 DB 연결

운영 PostgreSQL과 Neo4j는 SSH 터널로 접속합니다. 터널 명령과 계정은 `docs199-Server` 저장소의 `docs/ai-integration-setup.md`를 따르고, 비밀번호는 팀장에게 받습니다.

`DATABASE_URL`은 `postgresql://ai_app:비밀번호@localhost:5432/docs199` 형식으로 적습니다. PostgreSQL 연결은 풀로 재사용하며 최대 크기는 `DB_POOL_MAX_SIZE`(기본 10)입니다. `ai_app`은 `chunks`만 읽고 쓸 수 있고 나머지 허용 테이블은 읽기 전용이며, DDL을 실행할 수 없습니다. `chunks`는 1024차원이므로 `EMBEDDING_DIMENSIONS=1024`가 아니면 운영 저장소를 쓰지 않습니다. 스키마는 `docs199-Server`의 Flyway(V11, V12)가 관리합니다.

## 테스트

`ruff check . && pytest`로 검증합니다. 테스트는 `.env`를 읽지 않고 가짜 키를 쓰며, 외부 네트워크 접속을 차단하므로 API 비용이 들지 않습니다. 차단된 접속 시도가 있으면 테스트 전체가 실패로 끝납니다.

- 브라우저 테스트(`tests/e2e`)는 Chrome이 있을 때만 실행됩니다.
- `xfail`로 표시된 테스트는 아직 고치지 않은 결함입니다. 결함을 고치면 테스트가 실패로 알려주므로 표시를 지웁니다.
