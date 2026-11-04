# 근거 기반 사내 문서 검색 RAG MVP

최신 팀 기획은 [프로젝트 기획서](docs/프로젝트_기획서.md)에 정리했습니다. JEV가 기존 근거 컨텍스트의 재사용 여부를 판단하고, 신규 검색이 필요하면 두 DB의 결과를 RRF로 융합하는 방향입니다. 아래 내용은 현재 데모의 동작 및 실행 방법입니다.

질문이 들어오면 검색 모드 선택이나 JEV 라우팅 없이 Qdrant와 Neo4j를 항상 함께 검색합니다. Qdrant 벡터 검색과 Neo4j 전문 검색의 순위를 RRF로 합친 뒤, 최종 상위 청크만 답변 생성 단계에 전달합니다.

Streamlit은 채팅 형태로 여러 턴을 이어갑니다. 각 턴마다 최신 질문으로 두 검색기를 다시 조회하고, Claude에는 대화 이력과 그 턴에서 검색된 근거 청크를 함께 전달합니다. Claude는 대화 이력을 질문 맥락에만 사용하고, 사실은 현재 턴의 검색 근거로만 답하도록 안내됩니다. “그럼”, “그 경우”처럼 앞 질문을 가리키는 후속 질문은 직전 사용자 질문을 검색어에 보태고, 독립된 새 질문은 현재 질문만 검색합니다. 답변 아래 **LLM에 전달한 검색 컨텍스트**를 펼쳐 실제 입력 근거를 확인할 수 있습니다.

## AI 처리 흐름

```text
질문 + 필터
  ├─ Qdrant: 질문 임베딩 기반 Vector Search
  └─ Neo4j: 질문 키워드 기반 Full-text Search
         ↓
      RRF 중복 제거 및 순위 융합
         ↓
      최종 상위 청크 선택
         ↓
      Claude 또는 로컬 Ollama 답변 생성 / 근거 발췌 요약
```

질문 자체는 청킹하지 않습니다. 원문 문서를 적재할 때만 LangChain `RecursiveCharacterTextSplitter`로 청크를 만들고, 질문은 두 검색기에 전달합니다. 후속 질문은 직전 질문을 검색어에 추가할 수 있습니다. 각 검색은 `status=유효`, `allowed_group_ids`, 문서 필터를 데이터베이스 쿼리에 적용합니다. 답변에는 출처 청크와 원문 문자 위치가 함께 표시됩니다.

현재 데모는 매 질문마다 두 DB를 다시 검색합니다. 이전 턴 컨텍스트만으로 답할 수 있는지 판별하거나 검색 결과를 재사용하는 기능은 아직 구현하지 않았습니다. 최신 기획에서는 이 판단을 JEV가 맡고, 신규 검색 분기에서 두 DB를 조회하도록 변경할 예정입니다.

## 구성

- **Neo4j**: 문서, 청크, 관계 및 전문 검색 인덱스
- **Qdrant**: 청크 벡터와 검색 시점 필터
- **Ollama 로컬 모델**: 임베딩 및 선택적 답변 생성
- **Streamlit**: 자연어 질문, 권한/문서 필터, 검색 결과와 근거 확인
- **LangChain**: 문서 청킹과 Ollama 모델 연결

기본 답변은 LLM 없이 상위 근거를 발췌해 보여줍니다. Claude를 선택하면 질문과 검색된 근거 청크가 Anthropic API로 전송됩니다. 임베딩은 로컬 Ollama에서 수행합니다. Anthropic은 자체 임베딩 모델을 제공하지 않으므로 Claude 키만으로 임베딩할 수 없습니다.

Claude 답변을 사용하려면 기존 `.env`에 다음 항목을 추가하고 앱을 재시작하세요. 키는 채팅이나 저장소에 올리지 마세요.

```dotenv
CHAT_PROVIDER=anthropic
ANTHROPIC_API_KEY=발급받은_키
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`pip install -r requirements.txt`로 추가 의존성을 설치해야 합니다. 로컬 답변을 쓰려면 `CHAT_PROVIDER=ollama`와 `CHAT_MODEL`을 설정하세요. 모델 호출에 실패하면 화면에 오류와 함께 근거 발췌를 표시합니다.

## 실행

Python 3.10 이상과 Docker Compose가 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
```

기존 `.env`가 있다면 보존하고 `.env.example`의 필요한 설정만 기존 파일에 추가하세요. `.env`를 새로 만들 때만 복사합니다.

```bash
cp .env.example .env
```

Ollama를 설치해 실행한 뒤 임베딩 모델을 준비합니다.

```bash
ollama pull nomic-embed-text
```

앱을 실행합니다.

```bash
streamlit run app.py
```

사이드바에서 **더미 문서 DB 적재**를 누르면 `data/demo_documents.json`의 문서를 청킹하여 Neo4j와 Qdrant에 넣습니다. 다시 실행하면 같은 ID 기준으로 갱신합니다. Neo4j Browser는 `http://localhost:7474`에서 열 수 있습니다. 로컬 Compose 기본 계정은 `neo4j` / `local-dev-password`입니다.

Notion 페이지 원문이나 수동 스냅샷 데이터는 저장소에 포함하지 않습니다. 로컬에서 직접 준비한 `data/notion_planning.json`이 있으면 `python -m rag_demo.ingest_notion_snapshot`으로 별도 적재할 수 있습니다. 문자 위치는 스냅샷 텍스트 기준이며 Notion 원본의 위치와 동일하다고 보장하지 않습니다. 작성일 필드는 원본 작성일이 아닌 스냅샷 수집일입니다. 이 방식은 자동 동기화가 아닙니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt 주소 |
| `NEO4J_USER` | `neo4j` | Neo4j 사용자 |
| `NEO4J_PASSWORD` | `local-dev-password` | 로컬 Compose 개발 비밀번호 |
| `NEO4J_DATABASE` | `neo4j` | 데이터베이스 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 주소 |
| `QDRANT_COLLECTION` | `internal_chunks` | 컬렉션 이름 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 로컬 Ollama 주소 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | 로컬 임베딩 모델 |
| `CHAT_MODEL` | 빈 값 | 답변 생성용 로컬 Ollama 모델. 비어 있으면 근거 발췌 |
| `CHAT_PROVIDER` | `ollama` | 답변 생성기: `ollama` 또는 `anthropic` |
| `ANTHROPIC_API_KEY` | 빈 값 | Claude API 키 |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude 답변 모델 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `420` / `60` | 문서 청킹 크기와 겹침(문자 수) |

## 검색 필터와 문서 관계

화면의 그룹, 문서 유형, 출처, 문서 ID 필터는 Vector Search와 Full-text Search 양쪽 DB 쿼리에 전달됩니다. 권한 그룹을 비우면 접근 허용 문서가 없도록 동작합니다. 이는 필터 적용 시연용 구조이며 실제 로그인이나 운영 권한 검증은 포함하지 않습니다.

더미 문서에 정의한 문서 관계는 Neo4j에 저장합니다. 검색 결과의 최종 근거 청크가 관계의 근거 청크와 일치할 때만 관계를 추가로 보여주며, 관계 자체를 답변 근거로 쓰지는 않습니다.

기획서의 실제 Notion/Obsidian/드라이브 동기화와 관리자 기능, 원문 클릭 하이라이트 및 그래프 시각화는 이 데모 범위에 포함되지 않습니다. 원문 오프셋과 관계 근거는 검색 화면에서 확인할 수 있습니다.
