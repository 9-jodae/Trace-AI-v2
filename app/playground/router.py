from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)


@router.get("/playground")
async def playground() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))
