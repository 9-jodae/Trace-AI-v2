import pytest
from conftest import chunk

from app.agent.evidence import MAX_EVIDENCE, EvidenceCandidate, ground_evidence, parse_answer
from app.core.errors import AiUnavailableError

TEXT = "앞부분입니다. 연차는 최대 15일입니다. 뒷부분입니다."
QUOTE = "연차는 최대 15일입니다."
CONTEXT = [
    {"id": "a", "text": "무관한 내용", "source": "다른 문서", "document_id": "d0", "version_id": "v0",
     "char_start": 0, "char_end": 6},
    {"id": "b", "text": TEXT, "source": "휴가 규정", "document_id": "d1", "version_id": "v1",
     "char_start": 100, "char_end": 100 + len(TEXT)},
]


def candidate(number, quote=QUOTE):
    return EvidenceCandidate(chunk=number, quote=quote)


class TestGroundEvidence:
    def test_position_is_computed_from_document_offset(self):
        [evidence] = ground_evidence([candidate(2)], CONTEXT)
        start = TEXT.index(QUOTE)
        assert evidence["char_start"] == 100 + start and evidence["char_end"] == 100 + start + len(QUOTE)
        assert evidence["chunk_id"] == "b" and evidence["document_id"] == "d1" and evidence["version_id"] == "v1"
        assert evidence["source"] == "휴가 규정" and evidence["quote"] == QUOTE

    def test_position_counts_code_points(self):
        text = "😀" + QUOTE
        [evidence] = ground_evidence([candidate(1)], [{**CONTEXT[1], "text": text}])
        assert evidence["char_start"] == 101

    def test_quote_not_in_chunk_is_dropped(self):
        assert ground_evidence([candidate(2, "연차는 최대 20일입니다.")], CONTEXT) == []

    def test_reworded_whitespace_is_dropped(self):
        assert ground_evidence([candidate(2, "연차는  최대 15일입니다.")], CONTEXT) == []

    def test_quote_in_other_chunk_is_dropped(self):
        assert ground_evidence([candidate(1)], CONTEXT) == []

    def test_blank_quote_is_dropped(self):
        assert ground_evidence([candidate(2, "  ")], CONTEXT) == []
        assert ground_evidence([candidate(2, "")], CONTEXT) == []

    def test_out_of_range_number_is_dropped(self):
        assert ground_evidence([candidate(0), candidate(-1), candidate(3)], CONTEXT) == []

    def test_missing_offset_keeps_quote_without_position(self):
        [evidence] = ground_evidence([candidate(1)], [chunk("c", TEXT)])
        assert evidence["quote"] == QUOTE and evidence["char_start"] is None and evidence["char_end"] is None

    def test_duplicates_are_removed(self):
        assert len(ground_evidence([candidate(2), candidate(2)], CONTEXT)) == 1

    def test_limit(self):
        quotes = ["앞", "부", "분", "연", "차"]
        assert len(ground_evidence([candidate(2, q) for q in quotes], CONTEXT)) == MAX_EVIDENCE


class TestParseAnswer:
    def test_returns_answer_and_grounded_evidence(self):
        response = '{"answer": "15일입니다.", "evidence": [{"chunk": 2, "quote": "%s"}]}' % QUOTE
        answer, evidence = parse_answer(response, CONTEXT)
        assert answer == "15일입니다." and [item["chunk_id"] for item in evidence] == ["b"]

    def test_code_fence_is_accepted(self):
        response = '```json\n{"answer": "답", "evidence": []}\n```'
        assert parse_answer(response, CONTEXT) == ("답", [])

    def test_missing_evidence_is_empty(self):
        assert parse_answer('{"answer": "답"}', CONTEXT) == ("답", [])

    def test_forged_quote_keeps_answer_only(self):
        response = '{"answer": "답", "evidence": [{"chunk": 2, "quote": "없는 문구"}]}'
        assert parse_answer(response, CONTEXT) == ("답", [])

    def test_plain_text_is_kept_without_evidence(self, caplog):
        assert parse_answer("그냥 답변", CONTEXT) == ("그냥 답변", [])
        assert "그냥 답변" not in caplog.text

    def test_fenced_plain_text_is_kept_without_evidence(self):
        assert parse_answer("```\n그냥 답변\n```", CONTEXT)[1] == []

    @pytest.mark.parametrize(
        "response",
        [
            '{"answer": ""}',
            '{"answer": "   "}',
            '{"evidence": []}',
            '{"answer": "잘린 답',
            '```json\n{"answer": "  "}\n```',
        ],
    )
    def test_broken_json_is_not_shown_to_user(self, response):
        with pytest.raises(AiUnavailableError) as error:
            parse_answer(response, CONTEXT)
        assert "잘린" not in str(error.value)

    @pytest.mark.parametrize(
        "evidence",
        ['"문구"', "null", '[{"chunk": "2", "quote": "문구"}]', '[1, {"quote": "문구"}]'],
    )
    def test_malformed_evidence_keeps_answer(self, evidence):
        assert parse_answer(f'{{"answer": "답", "evidence": {evidence}}}', CONTEXT) == ("답", [])

    def test_malformed_item_keeps_other_quotes(self):
        response = '{"answer": "답", "evidence": [{"chunk": "2", "quote": "x"}, {"chunk": 2, "quote": "%s"}]}' % QUOTE
        assert [item["quote"] for item in parse_answer(response, CONTEXT)[1]] == [QUOTE]

    def test_answer_whitespace_is_trimmed(self):
        assert parse_answer('{"answer": "  답  "}', CONTEXT)[0] == "답"
