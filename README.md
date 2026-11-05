# docs199-AI

근거를 제시하는 사내 문서 검색 도우미의 AI 기획과 개발을 관리하는 레포입니다.

Python AI 서버는 FastAPI와 LangGraph로 검색과 답변 생성을 처리합니다.

이 저장소는 `9-jodae/docs199-AI`에서 옮겨 왔습니다. 자세한 내용은 [저장소 이전 기록](docs/repo-migration.md)을 참고하세요.

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
