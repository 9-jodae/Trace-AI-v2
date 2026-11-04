# docs199 AI 서버

사내 문서 검색 서비스의 Python AI 서버입니다. FastAPI와 LangGraph로 검색과 답변 생성을 처리합니다.

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
