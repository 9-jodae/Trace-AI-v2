from uuid import UUID

import pytest

from app.retrieval.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document

KOREAN_DOC = "재택근무 규정은 인사팀이 소유한다. 보안팀이 승인했다. " * 60


def test_short_text_is_one_chunk():
    chunks = chunk_document("doc", "ver", "짧은 글", "s")
    assert [c["text"] for c in chunks] == ["짧은 글"]
    assert chunks[0]["char_start"] == 0 and chunks[0]["char_end"] == 4


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_blank_text_has_no_chunks(text):
    assert chunk_document("doc", "ver", text, "s") == []


def test_chunks_respect_size_limit():
    chunks = chunk_document("doc", "ver", KOREAN_DOC, "s")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_chunk_ids_are_uuid_and_unique():
    ids = [c["id"] for c in chunk_document("doc", "ver", KOREAN_DOC, "규정")]
    assert all(UUID(chunk_id) for chunk_id in ids)
    assert len(set(ids)) == len(ids)


def test_chunk_ids_change_with_version_only():
    first = [c["id"] for c in chunk_document("doc", "v1", KOREAN_DOC, "s")]
    again = [c["id"] for c in chunk_document("다른문서", "v1", KOREAN_DOC, "다른출처")]
    other = [c["id"] for c in chunk_document("doc", "v2", KOREAN_DOC, "s")]
    assert first == again
    assert not set(first) & set(other)


def test_contract_fields_are_filled():
    chunk = chunk_document("doc", "ver", KOREAN_DOC, "규정")[0]
    assert chunk["document_id"] == "doc" and chunk["version_id"] == "ver" and chunk["source"] == "규정"


@pytest.mark.parametrize("text", [KOREAN_DOC, "".join(f"{i}번 문장입니다. " for i in range(300)), "가" * 3000])
def test_positions_point_at_chunk_text(text):
    chunks = chunk_document("doc", "ver", text, "s")
    assert chunks
    assert all(text[c["char_start"] : c["char_end"]] == c["text"] for c in chunks)


def test_same_input_gives_same_chunks():
    assert chunk_document("doc", "ver", KOREAN_DOC, "s") == chunk_document("doc", "ver", KOREAN_DOC, "s")


def test_splits_at_korean_sentence_end():
    chunks = chunk_document("doc", "ver", KOREAN_DOC, "s")
    assert all(c["text"].rstrip().endswith("다.") for c in chunks)


def test_neighbors_overlap():
    # 같은 문장이 반복되면 겹침 없이도 통과하므로 문장마다 내용을 다르게 만든다
    text = "".join(f"{i}번 항목을 설명하는 문장입니다. " for i in range(200))
    first, second = chunk_document("doc", "ver", text, "s")[:2]
    assert second["text"][:15] in first["text"]
    assert second["text"][:15] in first["text"][-CHUNK_OVERLAP:]


def test_prefers_paragraph_boundary():
    paragraph_a = "가" * 500
    paragraph_b = "나" * 500
    chunks = chunk_document("doc", "ver", f"{paragraph_a}\n\n{paragraph_b}", "s")
    assert [c["text"].strip() for c in chunks] == [paragraph_a, paragraph_b]


def test_text_without_separator_is_force_split():
    chunks = chunk_document("doc", "ver", "가" * (CHUNK_SIZE * 3), "s")
    assert len(chunks) >= 3
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_no_content_is_lost():
    sentences = [f"문장 번호 {i}번입니다. " for i in range(300)]
    joined = "".join(c["text"] for c in chunk_document("doc", "ver", "".join(sentences), "s"))
    assert all(s.strip() in joined for s in sentences)


def test_chunks_have_uuid_ids_and_source():
    chunks = chunk_document("doc", "ver", "가나다라. " * 400, "문서")

    assert len(chunks) > 1
    assert all(UUID(chunk["id"]) for chunk in chunks)
    assert all(chunk["source"] == "문서" for chunk in chunks)
    assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
