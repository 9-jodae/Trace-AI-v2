from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.graph import build_agent
from app.agent.router import router as agent_router
from app.core.errors import register_error_handlers
from app.playground.router import router as playground_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    yield


app = FastAPI(title="9-jodae Agent API", lifespan=lifespan)
register_error_handlers(app)

app.include_router(agent_router)
app.include_router(playground_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
