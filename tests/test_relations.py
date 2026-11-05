from app.retrieval.chunking import chunk_document
from app.retrieval.relations import document_names, resolve_references


def document(document_id, source, text):
    return {"document_id": document_id, "version_id": f"{document_id}-v1", "source": source, "text": text}


def test_document_names_include_title_and_first_nonempty_line():
    item = document("a", "a-file.pdf", "\n# 출장 신청 안내\n내용")
    assert document_names(item) == ["a-file.pdf", "출장 신청 안내"]


def test_resolves_only_explicit_known_references_to_source_chunk():
    source = document("a", "출장 신청 안내", "출장 신청 안내\n경비는 《경비 처리 절차》를 따릅니다.")
    target = document("b", "경비 처리 절차.pdf", "경비 처리 절차\n영수증을 첨부합니다.")
    chunks = chunk_document("a", "a-v1", source["text"], source["source"])

    links = resolve_references(source, chunks, [source, target])

    assert links == [{
        "source_document_id": "a",
        "target_document_id": "b",
        "chunk_id": chunks[0]["id"],
    }]


def test_supports_alternate_markup_and_deduplicates_target():
    source = document("a", "출발", "《도착》 [[도착]]")
    target = document("b", "도착", "도착")
    chunks = chunk_document("a", "a-v1", source["text"], source["source"])
    assert len(resolve_references(source, chunks, [source, target])) == 1


def test_ambiguous_unknown_and_self_references_are_skipped():
    source = document("a", "원본", "《중복 이름》 [[없는 문서]] 《원본》")
    first = document("b", "중복 이름", "첫 번째 문서")
    second = document("c", "중복 이름", "두 번째 문서")
    chunks = chunk_document("a", "a-v1", source["text"], source["source"])
    assert resolve_references(source, chunks, [source, first, second]) == []


def test_reference_match_is_normalized_and_does_not_guess_substring():
    source = document("a", "원본", "[[  경비   처리 절차 ]] [[경비 처리 절차서]]")
    target = document("b", "경비 처리 절차", "본문")
    chunks = chunk_document("a", "a-v1", source["text"], source["source"])
    links = resolve_references(source, chunks, [source, target])
    assert [link["target_document_id"] for link in links] == ["b"]
