# 로컬 실행

저장소를 복제한 뒤 `git config core.hooksPath .githooks`를 실행해 pre-push 훅을 활성화합니다. 훅은 원격 브랜치 삭제와 non-fast-forward push를 거부합니다. 이미 push한 브랜치는 rebase하지 말고 `git merge`로 갱신합니다.

1. Python 3.15로 가상 환경을 만들고 `pip install -r requirements-dev.txt`로 의존성을 설치합니다.
2. `.env.example`을 `.env`로 복사하고 값을 채웁니다. 각 항목의 설명은 `.env.example`에 있습니다.
3. `python main.py`를 실행합니다. 서버는 8000번 포트에서 시작합니다.

`DATABASE_URL`을 비워 두면 벡터 검색에 샘플 데이터를 사용합니다. 샘플 데이터를 쓸 때도 임베딩 API 키는 필요합니다. 답변을 생성하려면 `LLM_API_KEY`와 `LLM_MODEL`이 필요합니다.

대화 상태는 메모리에 저장하므로 서버를 재시작하면 사라집니다.
