# Python and FastAPI

- Keep HTTP mapping and input validation in `router.py`, request and response models in `schemas.py`, and feature logic in the feature package.
- Extend `CamelModel` for JSON models so fields match Spring's camelCase.
- Read configuration through `app/core/config.py`. Do not read environment variables elsewhere.
- Raise `AiUnavailableError` when search or the LLM cannot produce an answer. Return errors as `ErrorResponse` without internal exception text.
- Use parameterized SQL. Never build SQL or vector values by string concatenation with user input.
- Pin dependency versions in `requirements.txt` and development tools in `requirements-dev.txt`.
