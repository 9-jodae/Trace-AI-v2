import streamlit as st

from rag_demo.config import settings
from rag_demo.engine import run_search
from rag_demo.storage import LocalStores

st.set_page_config(page_title="근거 기반 사내 문서 검색", page_icon="🔎", layout="wide")
st.title("🔎 근거 기반 사내 문서 검색")
st.caption("대화하면서 질문하면 문서를 검색해 근거와 함께 답변합니다.")

with st.sidebar:
    st.subheader("검색 필터")
    allowed_groups = st.multiselect(
        "허용 그룹 (데모)",
        ["all-staff", "hr", "it-support"],
        default=["all-staff"],
        help="선택한 그룹의 문서만 두 검색 쿼리에 적용합니다. 빈 값이면 결과가 나오지 않습니다.",
    )
    doc_types = st.multiselect("문서 유형", ["규정", "업무 절차", "FAQ", "기획서"])
    sources = st.multiselect("문서 출처", ["local", "notion"], default=["local", "notion"])
    doc_ids_text = st.text_input("문서 ID 필터 (선택)", placeholder="POL-HR-001, FAQ-IT-001")
    top_k = st.slider("최종 근거 수", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("로컬 서비스")
    st.write(f"Neo4j: `{settings.neo4j_uri}`")
    st.write(f"Qdrant: `{settings.qdrant_url}`")
    st.write(f"임베딩: `{settings.embedding_model}`")
    if settings.chat_provider == "anthropic":
        st.write(f"답변 모델: `Claude / {settings.anthropic_model}`")
        st.caption("질문, 최근 대화와 검색된 근거가 Anthropic API로 전송됩니다.")
    else:
        st.write(f"답변 모델: `{settings.chat_model or '미설정 (근거 요약 모드)'}`")
    if st.button("연결 확인"):
        stores = None
        try:
            stores = LocalStores()
            for name, result in stores.check_connections().items():
                (st.success if result.startswith("연결됨") else st.error)(f"{name}: {result}")
        except Exception as exc:
            st.error(f"서비스 연결 점검 실패: {exc}")
        finally:
            if stores:
                stores.close()
    if st.button("더미 문서 DB 적재"):
        stores = None
        try:
            stores = LocalStores()
            count = stores.ingest_demo_documents()
            st.success(f"문서 청크 {count}개를 Neo4j와 Qdrant에 적재했습니다.")
        except Exception as exc:
            st.error(f"적재 실패: {exc}")
        finally:
            if stores:
                stores.close()

filters = {}
if doc_types:
    filters["doc_type"] = doc_types
if sources:
    filters["source"] = sources
if doc_ids_text.strip():
    filters["doc_id"] = [item.strip() for item in doc_ids_text.split(",") if item.strip()]

signature = (tuple(sorted(allowed_groups)), tuple(sorted(doc_types)), tuple(sorted(sources)), tuple(sorted(filters.get("doc_id", []))))
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("filter_signature", signature) != signature:
    st.session_state.messages = []
    st.info("검색 필터가 변경되어 이전 대화가 초기화되었습니다.")
st.session_state.filter_signature = signature

with st.sidebar:
    if st.button("새 대화 시작"):
        st.session_state.messages = []
        st.rerun()


def show_answer(item):
    st.caption(item.get("answer_mode", ""))
    st.markdown(item["content"])
    result = item.get("result")
    if not result:
        return
    context = result.get("llm_context", "")
    if context:
        with st.expander(f"LLM에 전달한 검색 컨텍스트 ({len(result['hits'])}개 청크)"):
            st.caption("이번 질문의 답변에 실제로 사용하도록 전달한 검색 근거입니다.")
            st.code(context, language=None)
    with st.expander(f"검색 결과와 출처 ({len(result['hits'])}건)"):
        st.caption(f"검색어: {result['search_query']} | 검색기: Qdrant Vector + Neo4j Full-text | RRF top-k: {result['top_k']}")
        for source, hits in result["raw_by_source"].items():
            st.write(f"{source}: {len(hits)}건")
        for index, hit in enumerate(result["hits"], start=1):
            st.markdown(f"**[{index}] {hit['title']}** · {hit['doc_id']} · {hit['chunk_id']}")
            st.caption(f"위치: {hit['char_start']}–{hit['char_end']} | 검색기: {', '.join(hit['methods'])} | RRF: {hit['score']:.5f}")
            if hit.get("source_url"):
                st.link_button("원본 기획서 열기", hit["source_url"], key=f"source-{item['id']}-{index}")
            st.code(hit["text"], language=None)
        for rel in result["graph_relations"]:
            st.write(f"관계: {rel['source_title']} → {rel['target_title']}: {rel['claim']} (근거: {rel['evidence_chunk_id']})")


for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant":
            show_answer(item)
        else:
            st.markdown(item["content"])

question = st.chat_input("문서에 관해 질문하세요")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        stores = None
        try:
            with st.spinner("관련 문서를 검색하고 답변을 작성하고 있습니다..."):
                stores = LocalStores()
                result = run_search(stores, question, filters=filters, allowed_group_ids=allowed_groups, top_k=top_k, history=history)
            item = {"role": "assistant", "content": result["answer"], "answer_mode": result["answer_mode"], "result": result, "id": len(st.session_state.messages)}
            st.session_state.messages.append(item)
            show_answer(item)
        except Exception as exc:
            item = {"role": "assistant", "content": f"검색 실패: {exc}", "error": True}
            st.session_state.messages.append(item)
            st.error(item["content"])
        finally:
            if stores:
                stores.close()
