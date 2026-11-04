from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.graph import build_agent
from app.agent.router import router as agent_router
from app.agent.store import ConversationStore
from app.core import config
from app.core.errors import register_error_handlers
from app.playground.router import router as playground_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    app.state.conversations = ConversationStore()
    yield


def create_app() -> FastAPI:
    config.validate_startup()
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
    if not production:
        app.include_router(playground_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
