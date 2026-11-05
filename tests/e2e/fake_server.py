"""가짜 AI로 실제 페이지를 띄운다. 외부 호출 없이 브라우저 동작만 확인한다."""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]

import safety

safety.install()

import uvicorn
from fastapi.responses import HTMLResponse

import main
from app.agent import graph
from app.core.errors import AiUnavailableError
from app.llm.client import NoDocumentReply

CHUNK = {"id": "a", "text": "재택근무 규정은 보안팀이 승인했다.", "source": "재택근무 규정"}
SCRIPT = (Path(__file__).parents[2] / "app/playground/index.html").read_text(encoding="utf-8")
probability = {"value": 0.92}


async def can_answer(question, context, history):
    if "JEV실패" in question:
        raise RuntimeError("x")
    return 0.1 if "새검색" in question else probability["value"]


async def search(query, filters=None, scope=None):
    if "전체실패" in query:
        raise AiUnavailableError("x")
    if "결과없음" in query:
        return {"chunks": [], "failed_sources": []}
    return {"chunks": [CHUNK], "failed_sources": ["graph"] if "부분실패" in query else []}


async def filter_context(chunks, scope=None):
    return chunks


async def generate(question, context, history):
    return f"<img src=x onerror=\"document.body.dataset.xss='1'\"> 가짜 답변: {question}"


async def small_talk_probability(question):
    return 0.0


async def generate_reply(question, allow_clarification=True):
    return NoDocumentReply(graph.NO_EVIDENCE_ANSWER, False)


async def rewrite_query(question):
    return None


graph.rewrite_search_query = rewrite_query
graph.small_talk_probability = small_talk_probability
graph.generate_reply_without_documents = generate_reply
graph.can_answer_probability = can_answer
graph.search = search
graph.filter_accessible_context = filter_context
graph.generate_answer = generate


@main.app.get("/__test", response_class=HTMLResponse)
def scenario(steps: str, reset_after: int = -1):
    # 질문을 순서대로 보내고 결과를 DOM에 남긴 뒤 표시한다
    questions = steps.split("|")
    return (
        SCRIPT
        + f"""<script>
(async () => {{
  const questions = {questions!r};
  const realFetch = window.fetch;
  const bodies = [];
  window.fetch = async (url, options) => {{ bodies.push(options.body); return realFetch(url, options); }};
  for (let i = 0; i < questions.length; i++) {{
    await run(questions[i]);
    if (i === {reset_after}) document.getElementById("reset").click();
  }}
  document.body.dataset.bodies = JSON.stringify(bodies);
  document.body.dataset.overflow = document.documentElement.scrollWidth > window.innerWidth ? "1" : "0";
  document.body.dataset.done = "1";
}})();
</script>"""
    )


uvicorn.run(main.app, port=int(sys.argv[1]), log_level="warning")
