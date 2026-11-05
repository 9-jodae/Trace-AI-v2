import re

from app.retrieval.schemas import CatalogDocument

# 문서·청크 수나 목록을 묻는 말. 주제 단어가 섞이면 내용 검색으로 보낸다
CATALOG_SUBJECT_PATTERN = re.compile(r"문서|자료|청크")
CATALOG_ASK_PATTERN = re.compile(r"몇|개수|목록|리스트|뭐뭐|종류|얼마나|총|전부|전체|무슨|어떤")
CATALOG_LIST_PATTERN = re.compile(r"목록|리스트|뭐뭐|종류|무슨|어떤|제목")
CATALOG_GENERIC_WORDS = (
    "ai",
    "시스템",
    "트레이스",
    "trace",
    "지금",
    "현재",
    "내가",
    "제가",
    "등록",
    "검색",
    "가능",
    "볼",
    "보는",
    "수",
    "있",
    "문서",
    "자료",
    "청크",
    "개수",
    "몇",
    "개",
    "건",
    "목록",
    "리스트",
    "뭐",
    "무엇",
    "무슨",
    "어떤",
    "어느",
    "종류",
    "얼마",
    "총",
    "전부",
    "전체",
    "모든",
    "다",
    "알려",
    "보여",
    "말해",
    "돼",
    "되",
    "나요",
    "니",
    "야",
    "는",
    "은",
    "이",
    "가",
    "에",
    "의",
    "들",
    "좀",
    "줘",
    "주세요",
    "된",
    "되어",
    "저장",
    "올라",
)
TYPE_LABELS = {"HANDOVER": "인수인계", "APPROVAL": "결재", "REPORT": "보고서", "PRESENTATION": "발표", "ETC": "기타"}
MAX_LISTED_TITLES = 30


def is_catalog_question(question: str) -> bool:
    if not CATALOG_SUBJECT_PATTERN.search(question) or not CATALOG_ASK_PATTERN.search(question):
        return False
    words = re.findall(r"[0-9A-Za-z가-힣%]+", question.lower())
    return all(word.startswith(CATALOG_GENERIC_WORDS) for word in words)


def describe_catalog(question: str, documents: list[CatalogDocument], filtered: bool) -> str:
    """DB에서 센 값으로 답한다. 호출자가 볼 수 있는 문서만 센다."""
    scope = "질문에 건 필터 안에서 " if filtered else ""
    if not documents:
        return f"{scope}지금 볼 수 있는 문서가 없습니다."
    chunks = sum(document["chunk_count"] for document in documents)
    lines = [f"{scope}지금 볼 수 있는 문서는 **{len(documents)}개**이고, 검색에 쓰는 청크는 **{chunks}개**입니다."]
    by_type: dict[str, int] = {}
    for document in documents:
        label = TYPE_LABELS.get(document["document_type"] or "", "미분류")
        by_type[label] = by_type.get(label, 0) + 1
    ordered = sorted(by_type.items(), key=lambda item: (-item[1], item[0]))
    lines.append("- 유형별: " + ", ".join(f"{label} {count}개" for label, count in ordered))
    if CATALOG_LIST_PATTERN.search(question):
        titles = [document["title"] for document in documents[:MAX_LISTED_TITLES]]
        lines.append("- 문서(제목순): " + ", ".join(titles))
        rest = len(documents) - len(titles)
        if rest > 0:
            lines.append(f"- 나머지 {rest}개는 문서 관리 화면에서 볼 수 있습니다.")
    return "\n".join(lines)
