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

실제 저장소가 아직 연결되지 않아, 샘플 데이터를 켜지 않으면 검색 경로는 503을 반환합니다. 샘플 문서가 실제 답변의 근거로 섞이지 않게 하기 위한 동작입니다.

대화는 메모리에 저장하며 대화당 최근 메시지 6개와 컨텍스트만 보관합니다. 최대 1,000개 대화, 1시간 동안 보관하고 서버를 재시작하면 사라집니다.

임베딩과 LLM을 Bedrock으로 쓰면 API 키 없이 AWS 자격 증명과 `AWS_REGION`을 사용합니다.

## 운영 DB 연결

운영 PostgreSQL과 Neo4j는 SSH 터널로 접속합니다. 터널 명령과 계정은 `docs199-Server` 저장소의 `docs/ai-integration-setup.md`를 따르고, 비밀번호는 팀장에게 받습니다.

`DATABASE_URL`은 `postgresql://ai_app:비밀번호@localhost:5432/docs199` 형식으로 적습니다. `ai_app`은 지정된 테이블의 읽기 권한만 있고 DDL을 실행할 수 없습니다. 벡터 검색이 쓰는 `chunks` 테이블과 쓰기 권한은 `docs199-Server`의 Flyway 마이그레이션으로 추가되어야 합니다.

## 테스트

`ruff check . && pytest`로 검증합니다. 테스트는 `.env`를 읽지 않고 가짜 키를 쓰며, 외부 네트워크 접속을 차단하므로 API 비용이 들지 않습니다. 차단된 접속 시도가 있으면 테스트 전체가 실패로 끝납니다.

- 브라우저 테스트(`tests/e2e`)는 Chrome이 있을 때만 실행됩니다.
- `xfail`로 표시된 테스트는 아직 고치지 않은 결함입니다. 결함을 고치면 테스트가 실패로 알려주므로 표시를 지웁니다.
