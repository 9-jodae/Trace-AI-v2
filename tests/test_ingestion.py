import asyncio
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from app.core.errors import AiUnavailableError
from app.ingestion import parsers, service
from app.ingestion.schemas import IngestRequest

JOB = "11111111-1111-1111-1111-111111111111"
VERSION = "22222222-2222-2222-2222-222222222222"
BODY = {"jobId": JOB, "versionId": VERSION, "storageKey": "documents/a.md", "extension": "md"}


def request(**changes):
    return IngestRequest(**{**BODY, **changes})


class TestParsers:
    def test_markdown_keeps_text_unchanged(self):
        assert parsers.extract_text("MD", "# 제목\n본문 ".encode()) == "# 제목\n본문 "

    @pytest.mark.parametrize("extension", ["docx", "hwp"])
    def test_unsupported_format(self, extension):
        with pytest.raises(parsers.UnsupportedFormatError):
            parsers.extract_text(extension, b"x")

    def test_pdf_pages_are_joined_like_server_export(self, monkeypatch):
        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class Reader:
            def __init__(self, stream):
                self.pages = [Page(" 첫째 "), Page(None), Page("셋째")]

        monkeypatch.setattr(parsers, "PdfReader", Reader)
        assert parsers.extract_text("pdf", b"x") == "첫째\n\n\n\n셋째"


class Body:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.data


class TestDownload:
    @pytest.fixture
    def s3(self, monkeypatch):
        calls = []

        class Client:
            size = 3

            def get_object(self, Bucket, Key):
                calls.append((Bucket, Key))
                return {"ContentLength": self.size, "Body": Body(b"abc")}

        client = Client()
        client.calls = calls
        monkeypatch.setattr(service, "_s3_client", lambda: client)
        return client

    def test_missing_bucket_fails_without_client(self):
        with pytest.raises(service.IngestionError):
            service._download("a.md")

    def test_uses_bucket_and_default_prefix(self, s3, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "b")
        monkeypatch.delenv("S3_PREFIX")
        assert service._download("/documents/a.md") == b"abc"
        assert s3.calls == [("b", "uploads/documents/a.md")]

    def test_custom_prefix(self, s3, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "/data/x/")
        service._download("a.md")
        assert s3.calls == [("b", "data/x/a.md")]

    def test_blank_prefix_means_none(self, s3, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "")
        service._download("a.md")
        assert s3.calls == [("b", "a.md")]

    def test_too_large_file_is_rejected(self, s3, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "b")
        s3.size = service.MAX_FILE_BYTES + 1
        with pytest.raises(service.IngestionError):
            service._download("a.md")


class Vector:
    def __init__(self, version=None):
        self.version = version if version is not None else {"document_id": "doc", "title": "규정"}

    async def find_version(self, version_id):
        return self.version or None


class Graph:
    def __init__(self):
        self.references = []

    async def name_targets(self):
        return {"경비 절차": {"other"}}

    async def replace_references(self, document_id, links):
        self.references.append((document_id, links))


@pytest.fixture
def pipeline(monkeypatch):
    state = {"vector": Vector(), "graph": Graph(), "indexed": [], "data": "규정은 《경비 절차》를 따른다.".encode()}

    async def index(document):
        state["indexed"].append(document)
        return [{"id": "c", "char_start": 0, "char_end": 100}]

    monkeypatch.setattr(service, "get_real_stores", lambda: (state["vector"], state["graph"]))
    monkeypatch.setattr(service, "_download", lambda key: state["data"])
    monkeypatch.setattr(service, "index_document", index)
    return state


class TestIndex:
    def run(self, **changes):
        asyncio.run(service._index(request(**changes)))

    def test_document_uses_server_metadata(self, pipeline):
        self.run()
        document = pipeline["indexed"][0]
        assert document["document_id"] == "doc" and document["source"] == "규정" and document["version_id"] == VERSION

    def test_references_are_linked_from_graph_names(self, pipeline):
        self.run()
        [(document_id, [link])] = pipeline["graph"].references
        assert document_id == "doc" and link["target_document_id"] == "other"

    def test_stale_version_skips_references(self, pipeline, monkeypatch):
        async def skipped(document):
            return []

        monkeypatch.setattr(service, "index_document", skipped)
        self.run()
        assert pipeline["graph"].references == []

    def test_unknown_version_fails(self, pipeline):
        pipeline["vector"] = Vector(version={})
        with pytest.raises(service.IngestionError):
            self.run()

    def test_unsupported_format_fails_with_safe_message(self, pipeline):
        with pytest.raises(service.IngestionError, match="지원하지 않는"):
            self.run(extension="hwp")

    def test_blank_text_fails(self, pipeline):
        pipeline["data"] = b"   "
        with pytest.raises(service.IngestionError):
            self.run()


@pytest.fixture
def callbacks(monkeypatch):
    sent = []

    async def fake(job_id, status, error=None):
        sent.append((job_id, status, error))

    monkeypatch.setattr(service, "_send_result", fake)
    return sent


class TestService:
    def submit_and_wait(self, *requests):
        async def run():
            ingestion = service.IngestionService()
            for item in requests:
                ingestion.submit(item)
                await asyncio.gather(*ingestion._tasks)
            return ingestion

        return asyncio.run(run())

    def test_success_reports_processing_then_done(self, pipeline, callbacks):
        self.submit_and_wait(request())
        assert callbacks == [(JOB, "PROCESSING", None), (JOB, "DONE", None)]

    def test_failure_message_is_passed(self, pipeline, callbacks):
        self.submit_and_wait(request(extension="hwp"))
        assert callbacks[-1] == (JOB, "FAILED", "지원하지 않는 파일 형식입니다.")

    def test_unexpected_error_hides_detail(self, pipeline, callbacks, monkeypatch, caplog):
        async def boom(document):
            raise RuntimeError("비밀 본문")

        monkeypatch.setattr(service, "index_document", boom)
        self.submit_and_wait(request())
        status, error = callbacks[-1][1:]
        assert status == "FAILED" and "RuntimeError" in error and "비밀" not in error and "비밀" not in caplog.text

    def test_missing_stores_report_failure(self, callbacks, monkeypatch):
        def missing():
            raise AiUnavailableError("없음")

        monkeypatch.setattr(service, "get_real_stores", missing)
        self.submit_and_wait(request())
        assert callbacks[-1][1] == "FAILED"

    def test_timeout_reports_failure(self, pipeline, callbacks, monkeypatch):
        async def slow(document):
            await asyncio.sleep(1)

        monkeypatch.setattr(service, "index_document", slow)
        monkeypatch.setattr(service, "JOB_TIMEOUT_SECONDS", 0.01)
        self.submit_and_wait(request())
        assert callbacks[-1] == (JOB, "FAILED", "처리 시간 상한을 넘었습니다.")

    def test_finished_job_resends_result_without_reindexing(self, pipeline, callbacks):
        self.submit_and_wait(request(), request())
        assert len(pipeline["indexed"]) == 1
        assert callbacks[-1] == (JOB, "DONE", None) and len(callbacks) == 3

    def test_running_job_is_not_started_twice(self, pipeline, callbacks):
        async def run():
            ingestion = service.IngestionService()
            ingestion.submit(request())
            ingestion.submit(request())
            await asyncio.gather(*ingestion._tasks)

        asyncio.run(run())
        assert len(pipeline["indexed"]) == 1

    def test_concurrency_is_limited(self, pipeline, callbacks, monkeypatch):
        active = peak = 0

        async def slow(document):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return []

        monkeypatch.setattr(service, "index_document", slow)

        async def run():
            ingestion = service.IngestionService()
            for number in range(6):
                ingestion.submit(request(jobId=UUID(int=number)))
            await asyncio.gather(*ingestion._tasks)

        asyncio.run(run())
        assert peak == service.MAX_CONCURRENT_JOBS

    def test_old_jobs_are_forgotten(self, monkeypatch):
        monkeypatch.setattr(service, "MAX_REMEMBERED_JOBS", 2)
        ingestion = service.IngestionService()
        for job in "abc":
            ingestion._remember(job, "DONE", None)
        assert list(ingestion._results) == ["b", "c"]

    def test_close_cancels_running_jobs(self, pipeline, callbacks, monkeypatch):
        finished = []

        async def slow(document):
            await asyncio.sleep(10)
            finished.append(True)

        monkeypatch.setattr(service, "index_document", slow)

        async def run():
            ingestion = service.IngestionService()
            ingestion.submit(request())
            await asyncio.sleep(0)
            await ingestion.close()

        asyncio.run(run())
        assert finished == []


class TestCallback:
    @pytest.fixture
    def spring(self, monkeypatch):
        state = {"requests": [], "statuses": []}
        real_client = httpx.AsyncClient

        def handler(item):
            state["requests"].append(item)
            return httpx.Response(state["statuses"].pop(0) if state["statuses"] else 200)

        monkeypatch.setattr(
            service.httpx,
            "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        )
        monkeypatch.setenv("SPRING_BASE_URL", "http://localhost:18080/")
        monkeypatch.setenv("INTERNAL_SECRET", "fake-secret")
        return state

    def test_sends_contract_format(self, spring):
        asyncio.run(service._send_result(JOB, "FAILED", "사유"))
        [sent] = spring["requests"]
        assert str(sent.url) == f"http://localhost:18080/api/internal/ingestion-jobs/{JOB}/result"
        assert sent.headers["X-Internal-Secret"] == "fake-secret"
        assert sent.read().decode() == '{"status":"FAILED","errorMessage":"사유"}'

    def test_retries_then_succeeds(self, spring):
        spring["statuses"] = [503, 200]
        asyncio.run(service._send_result(JOB, "DONE"))
        assert len(spring["requests"]) == 2

    def test_gives_up_without_raising(self, spring, caplog):
        spring["statuses"] = [503] * service.CALLBACK_ATTEMPTS
        asyncio.run(service._send_result(JOB, "DONE"))
        assert "gave up" in caplog.text and "fake-secret" not in caplog.text

    def test_missing_url_makes_no_call(self, spring, monkeypatch, caplog):
        monkeypatch.setenv("SPRING_BASE_URL", "")
        asyncio.run(service._send_result(JOB, "DONE"))
        assert spring["requests"] == [] and "SPRING_BASE_URL" in caplog.text


class TestRoute:
    @pytest.fixture
    def app_client(self, monkeypatch):
        submitted = []
        monkeypatch.setattr(service.IngestionService, "submit", lambda self, item: submitted.append(item))
        with TestClient(main.create_app()) as client:
            client.submitted = submitted
            yield client

    def test_accepts_spring_format(self, app_client):
        response = app_client.post("/ingest", json=BODY)
        assert response.status_code == 202
        [item] = app_client.submitted
        assert str(item.job_id) == JOB and item.storage_key == "documents/a.md" and item.extension == "md"

    @pytest.mark.parametrize("change", [{"jobId": "x"}, {"versionId": None}, {"storageKey": ""}, {"extension": ""}])
    def test_invalid_body_is_rejected(self, app_client, change):
        assert app_client.post("/ingest", json={**BODY, **change}).status_code == 422
        assert app_client.submitted == []

    def test_production_requires_secret(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("INTERNAL_SECRET", "fake-secret")
        monkeypatch.setenv("USE_SAMPLE_DATA", "")
        with TestClient(main.create_app()) as client:
            assert client.post("/ingest", json=BODY).status_code == 401
            ok = client.post("/ingest", json={**BODY, "extension": "hwp"}, headers={"X-Internal-Secret": "fake-secret"})
            assert ok.status_code == 202
