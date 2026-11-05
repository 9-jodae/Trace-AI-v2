# 팀 서버 실행

AI 서버를 팀 서버(EC2)에서 venv로 실행하는 절차입니다. 아직 실행해 보지 않았으므로 실제 결과에 맞춰 고칩니다.

## 구조

Spring, PostgreSQL, Neo4j는 Docker Compose에 있고 AI 서버는 호스트의 venv에서 실행합니다. DB 포트는 호스트의 `127.0.0.1`에 열려 있으므로 AI 서버는 `localhost`로 접속합니다. Spring은 컨테이너 안에 있으므로 호스트의 AI 서버로 접속할 경로가 따로 필요합니다.

## 설치

서버에 맞는 Python을 먼저 확인합니다. 로컬 개발과 CI는 Python 3.15 기준입니다.

```bash
python3 --version
git clone https://github.com/9-jodae/docs199-AI-v2.git ~/docs199-AI
cd ~/docs199-AI
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 설정

`.env`를 만들고 `chmod 600 .env`로 권한을 잠급니다. 값은 저장소에 커밋하지 않습니다.

| 항목 | 값 | 비고 |
| --- | --- | --- |
| `APP_ENV` | `production` | `local`이 아니면 운영 모드 |
| `HOST` | Docker 기본 브리지의 게이트웨이 IP | **비우면 `0.0.0.0`으로 열리므로 반드시 지정** |
| `PORT` | `8000` | |
| `INTERNAL_SECRET` | Spring과 같은 값 | 없으면 시작 실패 |
| `DATABASE_URL` | `postgresql://ai_app:비밀번호@localhost:5432/docs199` | |
| `EMBEDDING_MODEL` | `bedrock:amazon.titan-embed-text-v2:0` | 인스턴스 역할 사용 |
| `EMBEDDING_DIMENSIONS` | `1024` | |
| `BEDROCK_REGION` | `ap-northeast-1` | Titan과 Haiku 호출 리전. 서버의 `AWS_REGION`은 `ap-southeast-1`이라 따로 지정 |
| `LLM_MODEL` | `bedrock:jp.anthropic.claude-haiku-4-5-20251001-v1:0` | 도쿄 리전 추론 프로필 |
| `NEO4J_URI`, `JEV_API_KEY` | 비움 | 중간 발표 범위 밖 |

`USE_SAMPLE_DATA`와 `ACCESS_*`, `EMBEDDING_PROXY_URL`은 넣지 않습니다. 운영 모드에서 샘플 설정은 시작 실패이고, 나머지는 무시합니다.

게이트웨이 IP는 다음처럼 확인합니다.

```bash
docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'
```

Spring을 연결하기 전 시험에는 `HOST=127.0.0.1`을 써서 서버 안에서만 접속하게 합니다.

## 실행

재부팅 후에도 실행되도록 `/etc/systemd/system/docs199-ai.service`에 등록합니다.

```ini
[Unit]
Description=docs199 AI server
After=network-online.target docker.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/docs199-AI
ExecStart=/home/ubuntu/docs199-AI/.venv/bin/python main.py
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
curl -s -o /dev/null -w '%{http_code}\n' -X POST "http://<HOST 값>:8000/api/v1/agent/chat" \
  -H 'Content-Type: application/json' -d '{"message":"연결 확인"}'
```

외부 PC에서 `http://<서버 공인 IP>:8000`에 접속되면 안 됩니다. 접속되면 바로 서버를 멈추고 `HOST`와 보안 그룹을 확인합니다.

## Spring 연결

`docs199-Server`의 `docs/ai-integration-setup.md` "AI 서버 연결" 절을 따릅니다. Spring 컨테이너는 `http://host.docker.internal:8000`으로 호스트의 AI 서버에 접속합니다. 이 주소가 위의 브리지 게이트웨이를 가리키는지, 외부 인터페이스에서 8000 포트가 열리지 않는지 확인합니다.

서버의 `docs199-Server` 폴더에서 전환하고 되돌립니다. 스크립트가 `.env`의 `AI_MODE`, `AI_BASE_URL`을 바꾸고 잠금 배포까지 실행합니다.

```bash
scripts/switch-ai-mode.sh http http://host.docker.internal:8000
scripts/switch-ai-mode.sh mock
```

`INTERNAL_SECRET`은 Spring과 AI 서버가 같은 값이어야 합니다.

## 갱신과 중단

```bash
cd ~/docs199-AI && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart docs199-ai
sudo systemctl stop docs199-ai
```

문서를 다시 적재할 때는 서버의 JSON을 쓰고, 인스턴스 역할로 Titan을 직접 호출합니다.

```bash
.venv/bin/python -m app.retrieval.indexing ~/exports/demo-documents.json
```
