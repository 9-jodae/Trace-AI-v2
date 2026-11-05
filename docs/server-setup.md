# 팀 서버 실행

AI 서버를 팀 서버(EC2)에서 venv로 실행하는 절차입니다. 2026-11-05에 `dev` `8cb00d2`를 이 구성으로 올려 Python 3.12에서 질문, 근거, JEV 재사용이 동작하는 것을 확인했습니다.

## 구조

Spring, PostgreSQL, Neo4j는 Docker Compose에 있고 AI 서버는 호스트의 venv에서 실행합니다. DB 포트는 호스트의 `127.0.0.1`에 열려 있으므로 AI 서버는 `localhost`로 접속합니다. Spring은 컨테이너 안에 있으므로 호스트의 AI 서버로 접속할 경로가 따로 필요합니다.

## 설치

저장소가 비공개라 서버에 토큰을 두지 않고, 내 PC에서 압축해 복사합니다. 코드 폴더는 `~/docs199-AI-v2`입니다.

```bash
# 내 PC, 저장소 폴더에서
git fetch origin
git archive --format=tar.gz -o docs199-ai.tar.gz origin/dev
scp -i <키> docs199-ai.tar.gz ubuntu@<서버 공인 IP>:~/

# 서버에서
mkdir -p ~/docs199-AI-v2 && tar -xzf ~/docs199-ai.tar.gz -C ~/docs199-AI-v2
cd ~/docs199-AI-v2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 설정

설정은 `~/.config/docs199-ai/env`에 두고 `chmod 600`으로 권한을 잠급니다. systemd가 이 파일을 읽습니다. 값은 저장소에 커밋하지 않습니다.

| 항목 | 값 | 비고 |
| --- | --- | --- |
| `APP_ENV` | `production` | `local`이 아니면 운영 모드 |
| `INTERNAL_SECRET` | Spring과 같은 값 | 없으면 시작 실패 |
| `DATABASE_URL` | `postgresql://ai_app:비밀번호@localhost:5432/docs199` | |
| `EMBEDDING_MODEL` | `bedrock:amazon.titan-embed-text-v2:0` | 인스턴스 역할 사용 |
| `EMBEDDING_DIMENSIONS` | `1024` | |
| `BEDROCK_REGION` | `ap-northeast-1` | Titan과 답변 모델의 호출 리전. 서버의 기본 리전은 `ap-southeast-1`이고 그곳에는 Titan이 없어 따로 지정 |
| `LLM_MODEL` | `bedrock:global.anthropic.claude-sonnet-5-5` | 문서 답변 모델. `global.` 프로필은 요청을 다른 리전에서 처리할 수 있음 |
| `CHAT_LLM_MODEL` | `bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0` | 인사처럼 문서 없이 응대할 때 사용. 질문 문장만 전송. 전송 범위는 `docs199-Server#14`에서 합의 요청 중. 비우면 `LLM_MODEL` 사용 |
| `JEV_API_KEY` | TypeSafe 키 | 일상 대화 판단과 기존 컨텍스트 재사용 판단에 사용. 모든 질문이 TypeSafe로 전송되고 재사용 판단 때는 청크 본문도 전송됨 |
| `VECTOR_MAX_COSINE_DISTANCE` | `0.85` | 측정 근거는 로컬 실행 안내 참고 |
| `SPRING_INTERNAL_URL` | Spring 앱 내부 주소 | `/ingest` 결과 콜백 주소. 게이트웨이는 내부 경로를 막음. 운영 값은 `docs199-Server` 합의 대기 |
| `STORAGE_BUCKET`, `STORAGE_PREFIX` | Spring의 `S3_BUCKET`, `S3_PREFIX`와 같게 | `/ingest` 원본 읽기. 인스턴스 역할에 읽기 권한 필요 |
| `ENTITY_EXTRACTION_ENABLED` | `true` | 적재할 때 청크 본문을 `LLM_MODEL`로 보내 Entity와 관계를 추출. 끄고 재적재하면 그 문서의 Entity와 관계가 지워짐 |
| `DOCUMENT_TYPE_ENABLED` | `true` | `/ingest` 결과 콜백에 문서 유형을 담음. 제목과 앞부분 2,000자를 `LLM_MODEL`로 전송 |
| `NEO4J_URI` | 비움 | 값을 넣으면 인덱스가 없어 그래프 검색이 실패로 표시됨. PR #39와 Neo4j 적재 후 사용 |

`USE_SAMPLE_DATA`와 `ACCESS_*`, `EMBEDDING_PROXY_URL`은 넣지 않습니다. 운영 모드에서 샘플 설정은 시작 실패이고, 나머지는 무시합니다.

포트와 주소는 실행 명령에서 지정합니다. 게이트웨이 IP는 다음처럼 확인합니다.

```bash
docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'
```

`0.0.0.0`으로 열지 않습니다. AI 서버는 인증된 요청의 권한 값을 그대로 믿습니다.

## 실행

재부팅 후에도 실행되도록 `/etc/systemd/system/docs199-ai.service`에 등록합니다.

```ini
[Unit]
Description=docs199 AI server
After=network-online.target docker.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/docs199-AI-v2
EnvironmentFile=/home/ubuntu/.config/docs199-ai/env
ExecStart=/home/ubuntu/docs199-AI-v2/.venv/bin/uvicorn main:app --host 172.17.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now docs199-ai
journalctl -u docs199-ai -n 50 --no-pager
```

## 확인

비밀 헤더 없이 호출해 401이 나오면 서버와 인증이 동작하는 것입니다. 이 호출은 외부 API를 쓰지 않습니다.

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST "http://172.17.0.1:8000/api/v1/agent/chat" \
  -H 'Content-Type: application/json' -d '{"message":"연결 확인"}'
```

외부 PC에서 `http://<서버 공인 IP>:8000`에 접속되면 안 됩니다. 접속되면 바로 서버를 멈추고 실행 주소와 보안 그룹을 확인합니다.

## Spring 연결

`docs199-Server`의 `docs/ai-integration-setup.md` "AI 서버 연결" 절을 따릅니다. Spring 컨테이너는 `http://host.docker.internal:8000`으로 호스트의 AI 서버에 접속합니다. 이 주소가 위의 브리지 게이트웨이를 가리키는지, 외부 인터페이스에서 8000 포트가 열리지 않는지 확인합니다.

서버의 `docs199-Server` 폴더에서 전환하고 되돌립니다. 스크립트가 `.env`의 `AI_MODE`, `AI_BASE_URL`을 바꾸고 잠금 배포까지 실행합니다.

```bash
scripts/switch-ai-mode.sh http http://host.docker.internal:8000
scripts/switch-ai-mode.sh mock
```

`INTERNAL_SECRET`은 Spring과 AI 서버가 같은 값이어야 합니다.

## 갱신과 중단

위 설치처럼 압축 파일을 복사한 뒤 교체합니다. 폴더와 설정을 먼저 백업합니다. 재시작하는 몇 초 동안 Spring의 질문이 실패할 수 있습니다.

```bash
B=$(date +%H%M)
cp -a ~/docs199-AI-v2 ~/docs199-AI-v2.bak-$B
cp -p ~/.config/docs199-ai/env ~/.config/docs199-ai/env.bak-$B
tar -xzf ~/docs199-ai.tar.gz -C ~/docs199-AI-v2
cd ~/docs199-AI-v2 && .venv/bin/pip install -r requirements.txt
sudo systemctl restart docs199-ai
sudo systemctl stop docs199-ai
```

문서를 다시 적재할 때는 서버의 JSON을 쓰고, 인스턴스 역할로 Titan을 직접 호출합니다.

```bash
.venv/bin/python -m app.retrieval.indexing ~/exports/demo-documents.json
```
