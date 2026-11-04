import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.graph import build_agent
from app.agent.router import router as agent_router
from app.agent.store import ConversationStore
from app.core import config
from app.core.errors import register_error_handlers
from app.ingestion.router import router as ingestion_router
from app.jev.client import close_client as close_jev_client
from app.playground.router import router as playground_router
from app.retrieval.service import close_stores


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    app.state.conversations = ConversationStore(config.max_conversations(), config.conversation_ttl_seconds())
    yield
    await close_jev_client()
    await close_stores()


def configure_logging() -> None:
    # uvicorn은 앱 로거를 설정하지 않으므로 단계별 시간 로그가 보이도록 app 로거만 켠다
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        logger.addHandler(handler)


def create_app() -> FastAPI:
    config.validate_startup()
    configure_logging()
    production = config.is_production()
    app = FastAPI(
        title="9-jodae Agent API",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    register_error_handlers(app)
    app.include_router(agent_router)
    app.include_router(ingestion_router)
    if not production:
        app.include_router(playground_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", **config.server_options())
