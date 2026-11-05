import json

import pytest
from conftest import chunk

from app.agent.evidence import MAX_EVIDENCE, EvidenceCandidate, ground_evidence, parse_answer
from app.core.errors import AiUnavailableError

TEXT = "앞부분입니다. 연차는 최대 15일입니다. 뒷부분입니다."
QUOTE = "연차는 최대 15일입니다."
CONTEXT = [
    {
        "id": "a",
        "text": "무관한 내용",
        "source": "다른 문서",
        "document_id": "d0",
        "version_id": "v0",
        "char_start": 0,
        "char_end": 6,
    },
    {
        "id": "b",
        "text": TEXT,
        "source": "휴가 규정",
        "document_id": "d1",
        "version_id": "v1",
        "char_start": 100,
        "char_end": 100 + len(TEXT),
    },
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

    def test_surrounding_whitespace_is_trimmed_from_quote_and_position(self):
        [evidence] = ground_evidence([candidate(2, f" {QUOTE}\n")], CONTEXT)
        start = TEXT.index(QUOTE)
        assert evidence["quote"] == QUOTE
        assert evidence["char_start"] == 100 + start and evidence["char_end"] == 100 + start + len(QUOTE)

    def test_whitespace_variants_are_deduplicated(self):
        assert len(ground_evidence([candidate(2), candidate(2, f" {QUOTE}")], CONTEXT)) == 1

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
        answer, evidence = parse_answer(response, CONTEXT)[:2]
        assert answer == "15일입니다." and [item["chunk_id"] for item in evidence] == ["b"]

    def test_code_fence_is_accepted(self):
        response = '```json\n{"answer": "답", "evidence": []}\n```'
        assert parse_answer(response, CONTEXT)[:2] == ("답", [])

    def test_missing_evidence_is_empty(self):
        assert parse_answer('{"answer": "답"}', CONTEXT)[:2] == ("답", [])

    def test_forged_quote_keeps_answer_only(self):
        response = '{"answer": "답", "evidence": [{"chunk": 2, "quote": "없는 문구"}]}'
        assert parse_answer(response, CONTEXT)[:2] == ("답", [])

    def test_plain_text_is_kept_without_evidence(self, caplog):
        assert parse_answer("그냥 답변", CONTEXT)[:2] == ("그냥 답변", [])
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
            parse_answer(response, CONTEXT)[:2]
        assert "잘린" not in str(error.value)

    @pytest.mark.parametrize(
        "evidence",
        ['"문구"', "null", '[{"chunk": "2", "quote": "문구"}]', '[1, {"quote": "문구"}]'],
    )
    def test_malformed_evidence_keeps_answer(self, evidence):
        assert parse_answer(f'{{"answer": "답", "evidence": {evidence}}}', CONTEXT)[:2] == ("답", [])

    def test_malformed_item_keeps_other_quotes(self):
        response = '{"answer": "답", "evidence": [{"chunk": "2", "quote": "x"}, {"chunk": 2, "quote": "%s"}]}' % QUOTE
        assert [item["quote"] for item in parse_answer(response, CONTEXT)[1]] == [QUOTE]

    def test_parse_failure_log_has_no_content(self, caplog):
        with pytest.raises(AiUnavailableError):
            parse_answer('{"answer": "비밀 본문', CONTEXT)[:2]
        assert "LLM answer parse failed" in caplog.text and "비밀" not in caplog.text

    def test_no_evidence_answer_cannot_append_guesses(self):
        response = '{"answer": "관련 문서를 찾지 못했습니다. 인사팀에 문의하세요.", "evidence": []}'
        assert parse_answer(response, CONTEXT)[:2] == ("관련 문서를 찾지 못했습니다.", [])

    def test_partial_answer_with_grounded_quote_is_kept(self):
        answer = "관련 문서를 찾지 못했습니다. 다만 15일 기준은 있습니다."
        response = '{"answer": "%s", "evidence": [{"chunk": 2, "quote": "%s"}]}' % (answer, QUOTE)
        result, evidence = parse_answer(response, CONTEXT)[:2]
        assert result == answer and len(evidence) == 1

    def test_text_before_fenced_json_is_not_shown(self):
        body = '{"answer": "최종 답", "evidence": [{"chunk": 2, "quote": "%s"}]}' % QUOTE
        response = f"설명 문장입니다.\n\n```json\n{body}\n```"
        answer, evidence = parse_answer(response, CONTEXT)[:2]
        assert answer == "최종 답" and [item["quote"] for item in evidence] == [QUOTE]

    def test_text_around_json_without_fence_is_dropped(self):
        assert parse_answer('앞 {"answer": "답"} 뒤', CONTEXT)[:2] == ("답", [])

    def test_truncated_json_after_text_is_not_shown(self):
        with pytest.raises(AiUnavailableError):
            parse_answer('설명\n{"answer": "잘린', CONTEXT)[:2]

    def test_real_newline_in_answer_is_accepted(self):
        assert parse_answer('{"answer": "첫 줄\n둘째 줄"}', CONTEXT)[0] == "첫 줄\n둘째 줄"

    @pytest.mark.parametrize(
        "options, clarifying",
        [
            ('[{"label": "숙박비", "chunk": 1}, {"label": "할인", "chunk": 2}]', True),
            ('[{"label": "숙박비", "chunk": 1}]', False),
            ('[{"label": "가", "chunk": 1}, {"label": "나", "chunk": 1}]', False),
            ('[{"label": "가", "chunk": 1}, {"label": "나", "chunk": 9}]', False),
            ('[{"label": "가", "chunk": "1"}, {"label": "나", "chunk": 2}]', False),
            ('[{"label": " ", "chunk": 1}, {"label": "나", "chunk": 2}]', False),
            ("[]", False),
            ('"숙박비"', False),
        ],
    )
    def test_clarification_needs_two_distinct_chunks(self, options, clarifying):
        response = '{"answer": "어느 쪽인가요?", "options": %s}' % options
        assert parse_answer(response, CONTEXT).clarification is clarifying

    def test_clarification_clears_evidence(self):
        options = '[{"label": "가", "chunk": 1}, {"label": "나", "chunk": 2}]'
        response = '{"answer": "어느 쪽?", "evidence": [{"chunk": 2, "quote": "%s"}], "options": %s}' % (QUOTE, options)
        assert parse_answer(response, CONTEXT) == ("어느 쪽?", [], True, [])

    def test_refusal_wins_over_options(self):
        options = '[{"label": "가", "chunk": 1}, {"label": "나", "chunk": 2}]'
        response = '{"answer": "관련 문서를 찾지 못했습니다.", "options": %s}' % options
        assert parse_answer(response, CONTEXT) == ("관련 문서를 찾지 못했습니다.", [], False, [])

    def test_answer_whitespace_is_trimmed(self):
        assert parse_answer('{"answer": "  답  "}', CONTEXT)[0] == "답"


class TestKeywords:
    CONTEXT = [
        {"id": "a", "text": "미사용 연차는 최대 5일 이월한다.", "source": "문서"},
        {"id": "b", "text": "숙박비 한도", "source": "문서"},
    ]

    def parse(self, keywords):
        return parse_answer(json.dumps({"answer": "답", "evidence": [], "keywords": keywords}), self.CONTEXT)

    def test_keeps_only_text_found_in_chunk(self):
        keywords = [
            {"chunk": 1, "text": "최대 5일"},
            {"chunk": 2, "text": "최대 5일"},
            {"chunk": 2, "text": "지어낸 말"},
        ]
        result = self.parse(keywords)
        assert result.keywords == [{"chunk_id": "a", "text": "최대 5일"}]

    def test_limits_to_three_and_removes_duplicates(self):
        words = ["미사용", "연차", "연차", "최대", "이월"]
        result = self.parse([{"chunk": 1, "text": word} for word in words])
        assert [item["text"] for item in result.keywords] == ["미사용", "연차", "최대"]

    @pytest.mark.parametrize(
        "bad",
        [None, "연차", [{"chunk": "1", "text": "연차"}], [{"chunk": 9, "text": "연차"}], [{"chunk": 1, "text": " "}]],
    )
    def test_invalid_values_are_ignored(self, bad):
        assert self.parse(bad).keywords == []

    def test_long_text_is_ignored(self):
        long_text = "가" * 51
        context = [{"id": "a", "text": long_text, "source": "문서"}]
        response = json.dumps({"answer": "답", "keywords": [{"chunk": 1, "text": long_text}]})
        assert parse_answer(response, context).keywords == []

    def test_clarification_has_no_keywords(self):
        response = json.dumps(
            {
                "answer": "어느 쪽인가요?",
                "keywords": [{"chunk": 1, "text": "연차"}],
                "options": [{"label": "연차", "chunk": 1}, {"label": "숙박비", "chunk": 2}],
            }
        )
        assert parse_answer(response, self.CONTEXT).keywords == []


class TestInnerQuoteRepair:
    def test_unescaped_quotes_inside_answer_are_repaired(self):
        # 모델이 제목을 큰따옴표로 감싸고 이스케이프하지 않은 응답
        response = '{"answer": ""기준 정리 업무 가이드"를 요약하면 "핵심"입니다.", "evidence": []}'
        assert parse_answer(response, CONTEXT)[0] == '"기준 정리 업무 가이드"를 요약하면 "핵심"입니다.'

    def test_valid_json_is_unchanged(self):
        response = json.dumps({"answer": '그는 "네"라고 했다.', "evidence": []}, ensure_ascii=False)
        assert parse_answer(response, CONTEXT)[0] == '그는 "네"라고 했다.'

    def test_unrepairable_json_still_fails(self):
        with pytest.raises(AiUnavailableError):
            parse_answer('{"answer": "답", "evidence": [', CONTEXT)
