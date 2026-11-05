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

Upstage 임베딩을 테스트하려면 `.env`에 `APP_ENV=local`, `USE_SAMPLE_DATA=true`, `UPSTAGE_API_KEY=<키>`, `EMBEDDING_DIMENSIONS=1024`를 설정합니다. 질문과 샘플 문서가 Upstage로 전송되며 Solar Embedding 2 query/passage 모델을 각각 사용합니다. 기존 Titan 벡터와 혼합 검색하지 않도록 이 키는 실제 DB 모드에서 사용할 수 없습니다. 답변 생성에는 별도 LLM 설정이 필요합니다.

`DOCUMENT_TOPICS_ENABLED=true`이면 적재할 때 문서마다 제목과 앞부분 2,000자를 `LLM_MODEL`로 보내 주제를 1~3개 만들고 PostgreSQL `document_topics`에 저장합니다(`docs199-Server` V15 필요). 기본은 꺼짐이고 꺼져 있으면 이 테이블을 건드리지 않습니다. Spring은 `GRAPH_TOPIC_EDGES_ENABLED=true`일 때 저장된 주제로 문서 간선을 계산합니다. `python -m app.retrieval.document_topics <JSON 파일>`은 저장 없이 주제와 간선을 출력합니다. `NEO4J_URI`를 넣으면 질문 검색에도 Neo4j를 씁니다. 검색된 상위 문서가 본문에서 가리키는 문서를 최대 2단계 따라가, 질문과 가까운 청크를 최대 2개 더합니다. `GRAPH_SEARCH_ENABLED=false`면 끕니다.

검색은 PostgreSQL 벡터 검색과 전문 검색을 항상 함께 실행합니다. 두 결과의 상위 청크에서 문서 ID를 모아, 시작 문서도 PostgreSQL `accessible_chunks` 기반의 권한·현재 버전·문서 필터 검증을 통과한 경우에만 Neo4j 관계 탐색에 사용합니다. 이 하이브리드 시작점으로부터 문서의 명시적 `REFERENCES` 관계를 조회하고, 질문에 Entity가 직접 언급된 경우 기존 Entity 근거 청크 검색도 병행합니다. 그래프 후보는 PostgreSQL에서 다시 권한·현재 버전·필터를 검증하고 질문과 가까운 청크를 가져와 벡터·그래프·전문 검색 결과와 RRF로 결합합니다. 최종 상위 문서 3개에서 본문이 가리키는 문서를 최대 2단계 따라가고, 접근 가능한 관련 청크를 최대 2개 기존 결과 뒤에 더합니다. 따라서 현재 컨텍스트는 RRF 상위 5개에 연쇄 청크를 추가한 구성입니다.

관계 그래프는 문서 제목(파일명 확장자를 제외한 표기도 포함)에 이름이 명시된 문서를 대상으로 `《문서명》` 또는 `[[문서명]]` 표기를 추출합니다. 이름이 중복되거나 대상이 없으면 관계를 만들지 않습니다. JSON 일괄 적재는 입력 문서와 기존 Neo4j 문서 제목 목록을 함께 사용하므로 일부 문서만 적재해도 이미 등록된 다른 파일의 대상을 찾습니다. 문서 전체와 권한은 Neo4j에 복사하지 않으며 엔티티 관계의 근거 인용만 저장합니다. 문서 주제는 `DOCUMENT_TOPICS_ENABLED=true`일 때 제목과 앞부분 2,000자에서 1~3개를 `LLM_MODEL`로 생성해 PostgreSQL `document_topics`에 저장합니다(`docs199-Server` V15 필요). Spring은 `GRAPH_TOPIC_EDGES_ENABLED=true`일 때 저장된 주제로 문서 간선을 계산합니다. `python -m app.retrieval.document_topics <JSON 파일>`은 저장 없이 주제와 간선을 출력합니다.

질문 임베딩이 실패해도 전문 검색이 하이브리드 시작점을 제공하면 그래프 관계 경로를 계속 사용할 수 있습니다. 그래프 후보는 PostgreSQL에서 cosine distance 상한(기본 0.85)을 적용해 가져옵니다. 연쇄 청크는 거리 상한 없이 가까운 순으로 최대 2개를 기존 결과 뒤에 추가합니다. 전문 검색은 `simple` 설정이라 "연차는"처럼 조사가 붙은 단어는 "연차"와 맞지 않을 수 있습니다.

격리 PostgreSQL·Neo4j에서 앱 검색 API 전체 경로를 확인하려면 아래의 [격리 GraphDB 통합 테스트](#격리된-graphdb-통합-테스트)를 실행합니다. 기존 Desktop DB는 `Document.doc_id/title`, `Chunk.chunk_id` 구조인 반면 앱은 `Document.document_id/source`, `Chunk.id`, `REFERENCES` 관계를 기대하므로 그대로 연결해도 검색되지 않습니다. 또한 이 저장소의 PostgreSQL 접속 설정이 없으면 앱 검색은 503으로 종료됩니다. 팀 서버의 `NEO4J_URI`는 비워 두고, 실제 데이터 스키마와 PostgreSQL 권한 검증을 확인한 뒤에만 Neo4j를 켭니다.

벡터 검색은 pgvector cosine distance가 `VECTOR_MAX_COSINE_DISTANCE` 이하인 청크만 반환합니다. 기본값은 `0.85`입니다. 시연 문서 18개에 Titan v2로 질문 15개를 측정해 정했습니다. 관련 질문 9개의 정답 문서는 거리 0.435~0.821이었고, 무관 질문 6개의 1위는 0.827~0.950이었습니다. 표본이 작아 문서가 늘면 다시 재야 합니다. 맞는 문서를 놓치는 쪽이 더 나쁘다고 보아 여유를 두었고, 기준을 통과한 무관 질문은 답변 단계에서 거절합니다. 값은 0~2 범위여야 하고, 잘못된 값이면 서버가 시작되지 않습니다.

응답의 `answerType`은 `ANSWER`, `NO_EVIDENCE`, `CLARIFICATION`, `SMALL_TALK` 중 하나입니다. 이름과 값은 임시입니다(`docs199-Server#76`).

답변에는 핵심 문장 `evidence`와 핵심 키워드 `keywords`(`{chunkId, text}`)가 각각 최대 3개 붙습니다. 키워드도 원문에 그대로 있는 문구만 내보내고, 되묻기, 일상 대화, 거절에서는 빈 배열입니다. 형식은 `docs199-Server#78`을 따릅니다.

검색 결과가 없거나 찾은 문서로 답하지 못하면, 응대 모델이 "리브 팔러시"를 "leave policy"로 바꾸듯 다른 표기를 만들어 한 번 더 검색합니다. 다른 표기가 없으면 다시 검색하지 않습니다. 요청마다 JEV가 먼저 회사나 문서와 무관한 말인지 판단합니다. 확률이 0.9 이상이면 검색 없이 `CHAT_LLM_MODEL`이 답합니다. 판단에 실패하면 검색으로 진행합니다. 검색 결과가 없을 때도 같은 모델을 호출합니다. 인사, 메뉴 추천, 일반 상식, 도우미 기능 질문에는 답하고, 불분명한 문서 질문에는 되묻고, 회사 규정이나 사내 정보를 묻는 질문에는 고정 거절 문장을 씁니다. 이 되묻기도 앞의 되묻기와 같이 표시되어, 다음 말은 직전 질문을 붙여 검색하고 일상 대화 판단을 건너뛰며 연달아 되묻지 않습니다. 이 호출에는 대화 기록을 넘기지 않고, 응대 모델이 실패하면 `LLM_MODEL`로 넘어가며, 그것도 실패하면 거절 문장으로 답합니다.

질문이 애매하면 답변 LLM이 검색된 문서로 선택지를 만들어 되묻습니다. 서로 다른 대상이 2개 이상이고 질문과 대화로 좁힐 수 없을 때만 해당하며, 선택지가 서로 다른 청크를 2개 이상 가리켜야 인정합니다. 되물을 때는 `answer`에 "A에 관한 질문인가요, B에 관한 질문인가요? 좀 더 자세히 말씀해 주세요." 같은 문장이 담기고 `sources`와 `evidence`는 비어 있습니다. 연달아 되묻지 않으며, 다음 말은 직전 질문을 붙여 검색합니다.

권한은 서버의 `accessible_chunks`와 `filter_accessible_chunk_ids`로 판단합니다. 그래프 검색 결과와 재사용할 이전 근거도 `filter_accessible_chunk_ids`를 거치므로, PostgreSQL이 멈추면 그래프 결과도 내보내지 않습니다.

질문 요청의 `filters`는 서버의 9인자 `accessible_chunks`로 벡터 검색과 그래프 결과에 적용합니다. 필터가 있는 요청은 이전 근거를 재사용하지 않고 새로 검색합니다. 샘플 데이터에는 필터가 적용되지 않습니다.

검색 단계별 순위와 그래프 연쇄를 확인하려면 `/api/v1/agent/chat` 요청에 `"traceRetrieval": true`를 추가합니다. 응답의 `retrievalTrace`에는 검색 query, 벡터·전문·그래프 단계별 청크 순위, RRF 순위·점수·검색기별 순위, 그래프 seed와 권한 검증을 통과한 연쇄 문서/청크, 최종 출처 순위와 단계별 시간이 포함됩니다. 대화 컨텍스트를 재사용해 검색하지 않은 경우에는 `status: retrievalSkipped`로 표시됩니다. 문서 본문은 trace에 복제하지 않습니다. PostgreSQL에서 권한·현재 버전·요청 필터 검증을 통과해 가져온 청크만 trace에 나타납니다. Trace는 해당 요청 응답으로만 반환하며 서버 로그나 저장소에 남기지 않습니다.

```bash
curl -s http://localhost:8000/api/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"출장 영수증을 낸 다음 결재는 어떻게 진행돼?","traceRetrieval":true,"expectedDocumentIds":["travel-policy","expense-approval"],"accessScope":{"departmentId":"11111111-1111-4111-8111-111111111111","position":"STAFF","clearance":"INTERNAL","isAdmin":true}}'
```

`retrievalTrace.graph.neo4jCandidateOrder`는 PostgreSQL에서 권한·현재 버전·필터 검증을 통과한 Neo4j 후보 순서를 보존하고, `referenceChunkIds`와 `entityChunkIds`는 후보 경로를 구분합니다. `retrievalTrace.graph.chain.hops`는 단계별 `fromDocumentIds` → 권한 통과 `accessibleDocumentIds`를 보여줍니다. 선택적으로 `expectedDocumentIds`에 평가 정답 문서 ID를 넣으면 `retrievalTrace.evaluation.unexpectedCandidates`와 `unexpectedFinal`에 단계별·최종 기대 밖 청크 ID가 표시됩니다. 무관 질문은 빈 배열을 넣어야 모든 결과가 기대 밖으로 분류됩니다. 이 값은 평가 라벨로만 쓰이며 검색 필터나 권한 검사에 사용되지 않습니다. trace의 쿼리와 문서 ID는 권한 있는 요청자에게만 응답되므로, 진단 응답을 외부에 공유하기 전에 내부 식별자를 가리세요.

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

### 격리된 GraphDB 통합 테스트

실제 검색 경로를 확인하려면 Docker Desktop을 실행한 뒤 다음 명령을 사용합니다.

```bash
.venv/bin/python scripts/graph-integration/run.py
```

이 명령은 임시 PostgreSQL(pgvector)과 Neo4j를 별도 포트(15432, 17687)에 올리고, 무작위 ID의 가상 문서 2개를 실제 색인 코드로 적재한 다음 `POST /api/v1/agent/chat` 앱 경로를 호출합니다. 제목 없이 keyword-only·vector-only·혼합 seed 검색, 권한 거부, 오래된 버전 후보 차단과 최종 출처를 확인합니다. 실행이 끝나면 `docker compose down -v`로 컨테이너와 데이터를 정리합니다. 저장소의 `.env`와 연결된 Neo4j/PostgreSQL은 사용하지 않습니다.

격리 DB의 `accessible_chunks`는 이 검증에서 관리자 scope의 현재 버전만 반환하는 테스트용 함수입니다. 서버의 실제 권한 정책 검증을 대체하지 않습니다. 문서·질문 임베딩은 1024차원 결정적 대역이고 별도 사례에서 질문 임베딩 실패도 확인합니다. 답변 생성은 근거 인용을 검증할 수 있는 대역을 씁니다. 따라서 이 테스트는 Claude나 실제 임베딩 모델 호출을 검증하지 않습니다.
