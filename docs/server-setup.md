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
| `HOST` | Compose 네트워크의 게이트웨이 IP | **비우면 `0.0.0.0`으로 열리므로 반드시 지정** |
| `PORT` | `8000` | |
| `INTERNAL_SECRET` | Spring과 같은 값 | 없으면 시작 실패 |
| `DATABASE_URL` | `postgresql://ai_app:비밀번호@localhost:5432/docs199` | |
| `EMBEDDING_MODEL` | `bedrock:amazon.titan-embed-text-v2:0` | 인스턴스 역할 사용 |
| `EMBEDDING_DIMENSIONS` | `1024` | |
| `AWS_REGION` | `ap-northeast-1` | 중계 API와 같은 도쿄 리전 |
| `LLM_MODEL` | `bedrock:jp.anthropic.claude-haiku-4-5-20251001-v1:0` | 도쿄 리전 추론 프로필 |
| `NEO4J_URI`, `JEV_API_KEY` | 비움 | 중간 발표 범위 밖 |

`USE_SAMPLE_DATA`와 `ACCESS_*`, `EMBEDDING_PROXY_URL`은 넣지 않습니다. 운영 모드에서 샘플 설정은 시작 실패이고, 나머지는 무시합니다.

게이트웨이 IP는 다음처럼 확인합니다. 네트워크 이름은 `docker network ls`로 찾습니다.

```bash
docker network inspect <compose 네트워크> --format '{{(index .IPAM.Config 0).Gateway}}'
```

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

백엔드와 함께 진행합니다. `compose.prod.yaml`의 `app` 서비스에 다음이 필요합니다.

- `AI_MODE: http`. 지금은 `mock`으로 고정되어 있습니다.
- AI 서버 주소 설정. `AI_BASE_URL`을 `http://<게이트웨이 IP>:8000`으로 둡니다.
- `INTERNAL_SECRET`을 AI 서버와 같은 값으로 설정합니다.

서버 계약은 AI 서버를 같은 Docker 네트워크에 두는 구조로 적혀 있으므로, venv로 실행하는 동안의 차이를 백엔드와 공유합니다.

## 갱신과 중단

```bash
cd ~/docs199-AI && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart docs199-ai
sudo systemctl stop docs199-ai
```

문서를 다시 적재할 때는 서버의 JSON을 쓰고, 인스턴스 역할로 Titan을 직접 호출합니다.

```bash
.venv/bin/python -m app.retrieval.indexing ~/exports/demo-documents.json
```
