# Testing

- Test decision logic with unit tests in `tests/`. Replace external calls such as JEV, LLM, embedding, and database access with test doubles.
- Do not depend on `.env` or external services. Isolate state per test.
- Run the command in `docs/ai/workflows/verify.md`; report real results in Korean.
