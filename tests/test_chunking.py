import pytest

from app.retrieval.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document

KOREAN_DOC = "재택근무 규정은 인사팀이 소유한다. 보안팀이 승인했다. " * 60


def test_short_text_is_one_chunk():
    assert chunk_document("d", "짧은 글", "s") == [{"id": "d-1", "text": "짧은 글", "source": "s"}]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_blank_text_has_no_chunks(text):
    assert chunk_document("d", text, "s") == []


def test_chunks_respect_size_limit():
    chunks = chunk_document("d", KOREAN_DOC, "s")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_ids_are_sequential_and_source_is_kept():
    chunks = chunk_document("doc", KOREAN_DOC, "규정")
    assert [c["id"] for c in chunks] == [f"doc-{i}" for i in range(1, len(chunks) + 1)]
    assert {c["source"] for c in chunks} == {"규정"}


def test_same_input_gives_same_chunks():
    assert chunk_document("d", KOREAN_DOC, "s") == chunk_document("d", KOREAN_DOC, "s")


def test_splits_at_korean_sentence_end():
    chunks = chunk_document("d", KOREAN_DOC, "s")
    assert all(c["text"].rstrip().endswith("다.") for c in chunks)


def test_neighbors_overlap():
    # 같은 문장이 반복되면 겹침 없이도 통과하므로 문장마다 내용을 다르게 만든다
    text = "".join(f"{i}번 항목을 설명하는 문장입니다. " for i in range(200))
    first, second = chunk_document("d", text, "s")[:2]
    assert second["text"][:15] in first["text"]
    assert second["text"][:15] in first["text"][-CHUNK_OVERLAP:]


def test_prefers_paragraph_boundary():
    paragraph_a = "가" * 500
    paragraph_b = "나" * 500
    chunks = chunk_document("d", f"{paragraph_a}\n\n{paragraph_b}", "s")
    assert [c["text"].strip() for c in chunks] == [paragraph_a, paragraph_b]


def test_text_without_separator_is_force_split():
    chunks = chunk_document("d", "가" * (CHUNK_SIZE * 3), "s")
    assert len(chunks) >= 3
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_no_content_is_lost():
    sentences = [f"문장 번호 {i}번입니다. " for i in range(300)]
    joined = "".join(c["text"] for c in chunk_document("d", "".join(sentences), "s"))
    assert all(s.strip() in joined for s in sentences)


def test_chunks_have_sequential_ids_and_source():
    chunks = chunk_document("doc", "가나다라. " * 400, "문서")

    assert len(chunks) > 1
    assert [chunk["id"] for chunk in chunks] == [f"doc-{index}" for index in range(1, len(chunks) + 1)]
    assert all(chunk["source"] == "문서" for chunk in chunks)
    assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
