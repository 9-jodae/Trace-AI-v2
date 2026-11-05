import asyncio
import logging
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import main
from app.core import config
from app.core.errors import AiUnavailableError
from app.ingest import conflicts, service
from app.ingest.extraction import UnsupportedDocumentError, extract_text
from app.ingest.router import IngestTasks
from app.ingest.schemas import IngestRequest
from app.retrieval import indexing
from app.retrieval.chunking import chunk_document

SECRET_TEXT = "비밀 본문 내용"


def make_request(extension="md") -> IngestRequest:
    return IngestRequest(job_id=uuid4(), version_id=uuid4(), storage_key="documents/a.md", extension=extension)


def payload():
    return {"jobId": str(uuid4()), "versionId": str(uuid4()), "storageKey": "documents/a.md", "extension": "md"}


class VectorStore:
    def __init__(self, info=None, known=None, current=True):
        self.info = info
        self.known = known or {}
        self.current = current
        self.replaced = None
        self.hash_lookups = 0

    async def current_version_document(self, version_id):
        return self.info

    async def is_current_version(self, version_id):
        return self.current

    async def embeddings_by_hash(self, document_id, hashes):
        self.hash_lookups += 1
        return {key: value for key, value in self.known.items() if key in hashes}

    async def replace_version(self, version_id, chunks, embeddings):
        self.replaced = (chunks, embeddings)
        return True

    async def replace_document_topics(self, version_id, topics):
        raise AssertionError("주제는 켜지 않았다")


class GraphStore:
    def __init__(self):
        self.documents = []

    async def replace_document(self, document_id, source, names, chunks):
        self.documents.append((document_id, source, len(chunks)))


@pytest.fixture
def results(monkeypatch):
    sent = []

    async def send(job_id, status, error_message=None, conflict_results=None):
        sent.append((status, error_message))
        if conflict_results:
            sent.append(("CONFLICTS", conflict_results))

    monkeypatch.setattr(service, "send_result", send)
    return sent


def use_store(monkeypatch, store):
    monkeypatch.setattr(service, "get_real_stores", lambda: (store, GraphStore()))


def run(request):
    asyncio.run(service.process_ingest(request))


class TestEndpoint:
    @pytest.fixture
    def submitted(self, monkeypatch):
        seen = []
        monkeypatch.setattr(IngestTasks, "submit", lambda self, request: seen.append(request))
        return seen

    def test_accepts_spring_payload(self, submitted):
        with TestClient(main.app) as client:
            response = client.post("/ingest", json=payload())
        assert response.status_code == 202 and response.json() == {"accepted": True}
        assert submitted[0].storage_key == "documents/a.md"

    def test_rejects_invalid_payload(self, submitted):
        with TestClient(main.app) as client:
            assert client.post("/ingest", json={"jobId": "x"}).status_code == 422
        assert not submitted

    def test_requires_secret_in_production(self, monkeypatch, submitted):
        monkeypatch.setattr(config, "is_production", lambda: True)
        monkeypatch.setattr(config, "internal_secret", lambda: "s3cret")
        with TestClient(main.app) as client:
            assert client.post("/ingest", json=payload()).status_code == 401
            ok = client.post("/ingest", json=payload(), headers={"X-Internal-Secret": "s3cret"})
        assert ok.status_code == 202 and len(submitted) == 1


class TestTasks:
    def test_same_job_does_not_run_twice(self, monkeypatch):
        calls = []

        async def process(request):
            calls.append(request.job_id)
            await asyncio.sleep(0)

        monkeypatch.setattr("app.ingest.router.process_ingest", process)

        async def scenario():
            tasks = IngestTasks()
            request = make_request()
            tasks.submit(request)
            tasks.submit(request)
            tasks.submit(make_request())
            await asyncio.sleep(0.01)
            assert not tasks._tasks

        asyncio.run(scenario())
        assert len(calls) == 2


class TestExtraction:
    def test_text_is_kept_as_is(self):
        assert extract_text("MD", "  제목\n\n본문 ".encode()) == "  제목\n\n본문 "

    def test_pdf_without_text_is_empty(self):
        buffer = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.write(buffer)
        assert extract_text("pdf", buffer.getvalue()) == ""

    def test_unknown_extension(self):
        with pytest.raises(UnsupportedDocumentError):
            extract_text("hwp", b"x")


class TestProcess:
    def test_indexes_current_version(self, monkeypatch, results):
        store = VectorStore(info={"document_id": "d1", "title": "연차"})
        use_store(monkeypatch, store)
        captured = {}

        async def read(key):
            return "연차는 5일이다.".encode()

        async def index(document, **kwargs):
            captured.update(document=document, kwargs=kwargs)
            return [{"id": "c"}]

        monkeypatch.setattr(service, "read_object", read)
        monkeypatch.setattr(service, "index_document", index)
        request = make_request()
        run(request)
        assert results == [("PROCESSING", None), ("DONE", None)]
        assert captured["document"]["source"] == "연차" and captured["document"]["document_id"] == "d1"
        assert captured["document"]["version_id"] == str(request.version_id)
        assert captured["kwargs"] == {"reuse_embeddings": True}

    def test_old_version_is_skipped_without_reading_file(self, monkeypatch, results):
        use_store(monkeypatch, VectorStore(info=None))

        async def read(key):
            raise AssertionError("구버전 파일을 읽으면 안 된다")

        monkeypatch.setattr(service, "read_object", read)
        run(make_request())
        assert results[-1] == ("DONE", None)

    @pytest.mark.parametrize(
        "extension, content, message",
        [
            ("hwp", b"x", "지원하지 않는 문서 형식입니다."),
            ("md", b"   ", "문서 본문이 비어 있습니다."),
            ("md", b"\xff\xfe", "문서 본문을 추출하지 못했습니다."),
            ("pdf", b"not a pdf", "문서 본문을 추출하지 못했습니다."),
        ],
    )
    def test_bad_content_fails_with_fixed_message(self, monkeypatch, results, extension, content, message):
        use_store(monkeypatch, VectorStore(info={"document_id": "d1", "title": "t"}))

        async def read(key):
            return content

        monkeypatch.setattr(service, "read_object", read)
        run(make_request(extension))
        assert results[-1] == ("FAILED", message)

    def test_storage_failure_hides_details(self, monkeypatch, results, caplog):
        use_store(monkeypatch, VectorStore(info={"document_id": "d1", "title": "t"}))

        async def read(key):
            raise RuntimeError(SECRET_TEXT)

        monkeypatch.setattr(service, "read_object", read)
        run(make_request())
        assert results[-1] == ("FAILED", "원본 파일을 읽지 못했습니다.")
        assert SECRET_TEXT not in str(results)

    def test_index_failure_does_not_expose_content(self, monkeypatch, results, caplog):
        caplog.set_level(logging.INFO)
        use_store(monkeypatch, VectorStore(info={"document_id": "d1", "title": "t"}))

        async def read(key):
            return SECRET_TEXT.encode()

        async def index(document, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(service, "read_object", read)
        monkeypatch.setattr(service, "index_document", index)
        run(make_request())
        assert results[-1] == ("FAILED", "문서를 처리하지 못했습니다.")
        assert SECRET_TEXT not in caplog.text


class TestConflictRecheck:
    RELATION = "11111111-1111-4111-8111-111111111111"
    OTHER = "22222222-2222-4222-8222-222222222222"

    def request(self):
        return IngestRequest(
            job_id=uuid4(),
            version_id=uuid4(),
            storage_key="a.md",
            extension="md",
            conflicts=[{"relationId": self.RELATION, "item": "이월 일수", "otherDocumentId": self.OTHER}],
        )

    def run_with(self, monkeypatch, *, resolved=True, other=("상대", "상대 본문"), chunks=({"id": "c"},)):
        store = VectorStore(info={"document_id": "d1", "title": "연차"})
        store.other = other

        async def text(document_id):
            if isinstance(store.other, Exception):
                raise store.other
            return store.other

        store.current_document_text = text
        use_store(monkeypatch, store)

        async def read(key):
            return "새 본문".encode()

        async def index(document, **kwargs):
            return list(chunks)

        async def compare(*args):
            return resolved

        monkeypatch.setattr(service, "read_object", read)
        monkeypatch.setattr(service, "index_document", index)
        monkeypatch.setattr(service, "is_conflict_resolved", compare)
        run(self.request())

    def test_reports_result_with_done(self, monkeypatch, results):
        self.run_with(monkeypatch)
        assert ("CONFLICTS", [{"relationId": self.RELATION, "resolved": True}]) in results
        assert ("DONE", None) in results

    def test_missing_other_document_keeps_warning(self, monkeypatch, results):
        self.run_with(monkeypatch, other=None)
        assert ("CONFLICTS", [{"relationId": self.RELATION, "resolved": False}]) in results

    def test_read_failure_keeps_warning_and_finishes_job(self, monkeypatch, results):
        self.run_with(monkeypatch, other=RuntimeError("db down"))
        assert ("CONFLICTS", [{"relationId": self.RELATION, "resolved": False}]) in results
        assert ("DONE", None) in results

    def test_stale_version_is_not_compared(self, monkeypatch, results):
        self.run_with(monkeypatch, chunks=())
        assert not any(item[0] == "CONFLICTS" for item in results)

    def test_without_conflicts_callback_is_unchanged(self, monkeypatch, results):
        store = VectorStore(info={"document_id": "d1", "title": "연차"})
        use_store(monkeypatch, store)

        async def read(key):
            return "본문".encode()

        async def index(document, **kwargs):
            return [{"id": "c"}]

        monkeypatch.setattr(service, "read_object", read)
        monkeypatch.setattr(service, "index_document", index)
        run(make_request())
        assert results == [("PROCESSING", None), ("DONE", None)]


class TestConflictJudge:
    def judge(self, monkeypatch, reply, text="가", other="나"):
        async def respond(messages):
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(conflicts, "generate_structured_response", respond)
        return asyncio.run(conflicts.is_conflict_resolved("항목", "A", text, "B", other))

    def test_true_only_when_explicit(self, monkeypatch):
        assert self.judge(monkeypatch, '{"resolved": true}') is True
        assert self.judge(monkeypatch, '{"resolved": false}') is False
        assert self.judge(monkeypatch, '{"resolved": "true"}') is False

    def test_bad_reply_keeps_warning(self, monkeypatch):
        assert self.judge(monkeypatch, "그냥 글") is False
        assert self.judge(monkeypatch, AiUnavailableError("x")) is False

    def test_too_long_is_not_sent(self, monkeypatch):
        long_text = "가" * (conflicts.MAX_COMPARE_CHARACTERS + 1)
        assert self.judge(monkeypatch, '{"resolved": true}', text=long_text) is False


class TestEmbeddingReuse:
    def run_index(self, monkeypatch, store, reuse):
        embedded = []

        async def embed(texts):
            embedded.extend(texts)
            return [[float(len(text))] for text in texts]

        monkeypatch.setattr(indexing, "embed_documents", embed)
        monkeypatch.setattr(indexing, "get_real_stores", lambda: (store, GraphStore()))
        document = {"document_id": "d1", "version_id": "v2", "text": "가" * 700 + "\n\n" + "나" * 700, "source": "t"}
        chunks = chunk_document("d1", "v2", document["text"], "t")
        assert len(chunks) == 2

        async def references(*args):
            return None

        monkeypatch.setattr(indexing, "_replace_document_references", references)

        async def catalog(*args):
            return []

        monkeypatch.setattr(indexing, "_reference_catalog", catalog)
        asyncio.run(indexing.index_document(document, reuse_embeddings=reuse))
        return chunks, embedded

    def test_only_changed_chunks_are_embedded(self, monkeypatch):
        chunks = chunk_document("d1", "v2", "가" * 700 + "\n\n" + "나" * 700, "t")
        store = VectorStore(known={chunks[0]["content_hash"]: [9.0]})
        chunks, embedded = self.run_index(monkeypatch, store, True)
        assert embedded == [chunks[1]["text"]]
        assert store.replaced[1] == [[9.0], [float(len(chunks[1]["text"]))]]

    def test_default_does_not_reuse(self, monkeypatch):
        chunks = chunk_document("d1", "v2", "가" * 700 + "\n\n" + "나" * 700, "t")
        store = VectorStore(known={chunks[0]["content_hash"]: [9.0]})
        _, embedded = self.run_index(monkeypatch, store, False)
        assert len(embedded) == 2 and store.hash_lookups == 0


class TestCallback:
    def test_retries_then_gives_up_quietly(self, monkeypatch):
        attempts = []

        def handler(request):
            attempts.append(request)
            return httpx.Response(500)

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            service.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
        )
        monkeypatch.setattr(service, "CALLBACK_RETRY_SECONDS", 0)
        monkeypatch.setattr(config, "internal_secret", lambda: "s3cret")
        monkeypatch.setenv("SPRING_INTERNAL_URL", "http://spring:8080/")
        asyncio.run(service.send_result("job-1", "DONE"))
        assert len(attempts) == service.CALLBACK_ATTEMPTS
        assert str(attempts[0].url) == "http://spring:8080/api/internal/ingestion-jobs/job-1/result"
        assert attempts[0].headers["X-Internal-Secret"] == "s3cret"
