import html as html_lib
import json
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not CHROME.exists(), reason="Chrome 없음")]


@pytest.fixture(scope="module")
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("fake_server.py")), str(port)])
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{url}/playground", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    yield url
    process.terminate()
    process.wait()


def open_page(server, tmp_path, steps, width=900, **params):
    query = urllib.parse.urlencode({"steps": "|".join(steps), **params})
    result = subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            f"--window-size={width},1200",
            "--virtual-time-budget=8000",
            f"--user-data-dir={tmp_path}",
            "--dump-dom",
            f"{server}/__test?{query}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    dom = result.stdout
    assert 'data-done="1"' in dom, dom[-500:]
    return dom


def classes(dom, element_id):
    match = re.search(rf'id="{element_id}"[^>]*?class="([^"]*)"|class="([^"]*)"[^>]*?id="{element_id}"', dom)
    return (match.group(1) or match.group(2)).split()


def text_of(dom, element_id):
    match = re.search(rf'id="{element_id}"[^>]*>(.*?)</', dom, re.S)
    return html_lib.unescape(match.group(1))


def body_attribute(dom, name):
    return html_lib.unescape(re.search(rf'data-{name}="([^"]*)"', dom).group(1))


def test_first_question_goes_to_search(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문"])
    assert "on" in classes(dom, "n-vector") and "on" in classes(dom, "n-graph") and "on" in classes(dom, "n-rrf")
    assert "skip" in classes(dom, "n-jev") and "skip" in classes(dom, "n-context")
    assert "호출하지 않고" in text_of(dom, "reason")


def test_second_question_reuses_context(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문", "둘째 질문"])
    assert "on" in classes(dom, "n-jev") and "on" in classes(dom, "n-context") and "on" in classes(dom, "n-llm")
    assert "skip" in classes(dom, "n-vector") and "skip" in classes(dom, "n-rrf")
    assert "92%" in text_of(dom, "reason")


def test_low_probability_searches_again(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문", "새검색 질문"])
    assert "on" in classes(dom, "n-jev") and "on" in classes(dom, "n-vector")
    assert "10%" in text_of(dom, "reason")


def test_thread_id_is_kept_between_questions(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문", "둘째 질문"])
    bodies = [json.loads(body) for body in json.loads(body_attribute(dom, "bodies"))]
    assert bodies[0]["threadId"] is None and bodies[1]["threadId"]


def test_new_conversation_resets_thread(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문", "둘째 질문"], reset_after=0)
    bodies = [json.loads(body) for body in json.loads(body_attribute(dom, "bodies"))]
    assert bodies[1]["threadId"] is None
    assert "호출하지 않고" in text_of(dom, "reason")


def test_partial_failure_shows_notice(server, tmp_path):
    dom = open_page(server, tmp_path, ["부분실패 질문"])
    assert "skip" in classes(dom, "n-graph") and "on" in classes(dom, "n-vector")
    assert "graph" in text_of(dom, "notice") and "실패" in text_of(dom, "s-graph")


def test_total_failure_shows_error(server, tmp_path):
    dom = open_page(server, tmp_path, ["전체실패 질문"])
    assert "요청 실패" in text_of(dom, "notice") and "현재 답변을 생성할 수 없습니다" in text_of(dom, "notice")


def test_jev_failure_shows_reason(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문", "JEV실패 질문"])
    assert "JEV 호출 실패" in text_of(dom, "reason") and "on" in classes(dom, "n-vector")


def test_no_results_skips_llm(server, tmp_path):
    dom = open_page(server, tmp_path, ["결과없음 질문"])
    assert "skip" in classes(dom, "n-llm") and "관련 문서를 찾지 못했습니다" in text_of(dom, "answer")


def test_response_html_is_not_executed(server, tmp_path):
    dom = open_page(server, tmp_path, ["첫 질문"])
    assert 'data-xss' not in dom
    assert "<img" in text_of(dom, "answer") and '<img src="x"' not in dom


@pytest.mark.parametrize("width", [390, 900])
def test_no_horizontal_page_scroll(server, tmp_path, width):
    assert body_attribute(open_page(server, tmp_path, ["첫 질문"], width=width), "overflow") == "0"
