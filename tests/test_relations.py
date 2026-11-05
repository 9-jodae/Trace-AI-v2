import pytest

from app.retrieval.chunking import chunk_document
from app.retrieval.relations import document_names, hop_depth, name_in_query, normalize_name, resolve_references


def document(doc_id, title, text):
    return {"document_id": doc_id, "version_id": f"v-{doc_id}", "text": text, "source": title}


def chunks(item):
    return chunk_document(item["document_id"], item["version_id"], item["text"], item["source"])


TRAVEL = document("travel", "travel-policy", "출장비 규정\n결재는 《경비 결재 절차》 제3조를 따른다.")
EXPENSE = document("expense", "expense-approval", "# 경비 결재 절차\n본문")


def test_names_include_title_and_first_line():
    assert document_names(EXPENSE) == ["expense-approval", "경비 결재 절차"]


def test_normalize_collapses_case_and_spaces():
    assert normalize_name("  Hello   World ") == "hello world"


@pytest.mark.parametrize(
    "name, query, expected",
    [
        ("hr", "hr 담당", True),
        ("hr", "chrome", False),
        ("출장비 규정", "출장비 규정은", True),
        ("가", "가나다", False),
    ],
)
def test_name_in_query(name, query, expected):
    assert name_in_query(name, query) is expected


@pytest.mark.parametrize("query, depth", [("연결된 문서", 2), ("hr 관련", 2), ("연차 일수", 1)])
def test_hop_depth(query, depth):
    assert hop_depth(query) == depth


class TestResolveReferences:
    def test_links_quoted_document_with_position(self):
        [link] = resolve_references(TRAVEL, chunks(TRAVEL), [TRAVEL, EXPENSE])
        assert link["source_document_id"] == "travel" and link["target_document_id"] == "expense"
        assert TRAVEL["text"][link["char_start"] : link["char_end"]] == link["quote"] == "《경비 결재 절차》"
        assert link["chunk_id"] == chunks(TRAVEL)[0]["id"]

    def test_wiki_link_is_supported(self):
        item = document("map", "지도", "[[경비 결재 절차]]를 본다.")
        assert [x["target_document_id"] for x in resolve_references(item, chunks(item), [item, EXPENSE])] == ["expense"]

    def test_unknown_name_is_ignored(self):
        item = document("a", "제목", "《없는 문서》를 본다.")
        assert resolve_references(item, chunks(item), [item]) == []

    def test_self_reference_is_ignored(self):
        item = document("a", "제목", "《제목》을 본다.")
        assert resolve_references(item, chunks(item), [item]) == []

    def test_ambiguous_name_is_ignored(self):
        other = document("other", "경비 결재 절차", "다른 문서")
        assert resolve_references(TRAVEL, chunks(TRAVEL), [TRAVEL, EXPENSE, other]) == []

    def test_repeated_reference_links_once(self):
        item = document("a", "제목", "《경비 결재 절차》와 《경비 결재 절차》")
        assert len(resolve_references(item, chunks(item), [item, EXPENSE])) == 1

    def test_position_outside_chunks_is_ignored(self):
        assert resolve_references(TRAVEL, [], [TRAVEL, EXPENSE]) == []
