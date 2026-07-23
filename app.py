import json
import tempfile
import time
import os
from datetime import datetime, date

import streamlit as st
import streamlit.components.v1 as components

from config import settings, validate_required_keys
from modules.pdf_loader import extract_text_from_pdf, PDFExtractionError
from modules.chunker import chunk_pages
from modules.embedder import get_embedder
from modules.vector_store import PineconeVectorStore, VectorStoreError
from modules.retriever import Retriever
from modules.generator import GroqGenerator, estimate_confidence
from modules.logger import log_query, read_recent_logs, clear_logs, log_file_path
from modules.stt import transcribe_audio, STTError
from modules import store

st.set_page_config(page_title="Lexora", page_icon=":material/auto_awesome:", layout="wide")

# ----------------------------------------------------------------------
# Theme (Appearance, set from the Settings dialog) — CSS-variable based
# since Streamlit's native widget chrome can't be re-themed at runtime;
# this repaints everything this app itself draws (chat bubbles, sidebar,
# cards, source cards) plus the major stTestId containers.
# ----------------------------------------------------------------------
LIGHT = dict(bg="#ffffff", bg2="#f7f7f8", sidebar="#f7f7f8", ink="#1e1b2e",
             muted="#6b7280", border="#e5e7eb", card="#fafafa",
             brand="#4f46e5", brand_light="#eef2ff")
DARK = dict(bg="#212121", bg2="#171717", sidebar="#171717", ink="#ececec",
            muted="#9ca3af", border="#3a3a3a", card="#2a2a2a",
            brand="#818cf8", brand_light="#312e81")


def _vars(v: dict) -> str:
    return (
        f"--bg:{v['bg']};--bg2:{v['bg2']};--sidebar-bg:{v['sidebar']};"
        f"--ink:{v['ink']};--muted:{v['muted']};--border:{v['border']};"
        f"--card-bg:{v['card']};--brand:{v['brand']};--brand-light:{v['brand_light']};"
    )


def theme_vars_css(theme: str) -> str:
    css = f":root {{ {_vars(LIGHT)} }}\n"
    if theme == "dark":
        css = f":root {{ {_vars(DARK)} }}\n"
    elif theme == "system":
        css += f"@media (prefers-color-scheme: dark) {{ :root {{ {_vars(DARK)} }} }}\n"
    return css


if "theme" not in st.session_state:
    st.session_state.theme = store.load_settings().get("theme", "system")

st.markdown(
    f"""
    <style>
      {theme_vars_css(st.session_state.theme)}

      /* Deploy/Rerun/menu are already removed server-side via
         `toolbarMode = "viewer"` in .streamlit/config.toml — that's the
         stable, official switch for it. We deliberately do NOT touch
         [data-testid="stHeader"]/stToolbar with CSS visibility rules here:
         doing that guesses at internal test-ids that change between
         Streamlit versions, and on the last deploy it ended up hiding the
         "reopen sidebar" arrow too, with no way to get it back. Leaving
         the header alone guarantees that control keeps working. */
      #MainMenu, footer {{visibility: hidden;}}
      [data-testid="stHeader"] {{background: var(--bg) !important; box-shadow: none !important;}}
      .block-container {{padding-top: 1.5rem; padding-bottom: 6rem; max-width: 860px;}}

      /* The chat input is pinned to the bottom of the viewport outside the
         centered .block-container, so on a wide layout (or a closed
         sidebar) it stretches edge-to-edge instead of lining up with the
         rest of the page. Constrain and center it to match. */
      [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {{
        max-width: 860px !important; margin-left: auto !important; margin-right: auto !important;
      }}

      html, body, [class*="css"] {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; }}

      [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{
        background: var(--bg) !important;
      }}
      [data-testid="stSidebar"] {{ background: var(--sidebar-bg) !important; }}
      body, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
      .stMarkdown, h1, h2, h3, label, .stCaption {{ color: var(--ink); }}

      .app-header h1 {{ font-size: 1.8rem; font-weight: 700; margin: 0; color: var(--ink); }}
      .app-subtitle {{ color: var(--muted); font-size: 0.92rem; margin-top: -0.2rem; margin-bottom: 1rem; }}

      .stButton>button {{ border-radius: 8px; }}
      .stButton>button[kind="primary"] {{
        background: var(--brand); border: none; font-weight: 600;
      }}
      .stButton>button[kind="secondary"] {{
        background: transparent; border: 1px solid transparent; text-align: left; color: var(--ink);
      }}
      [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {{ background: var(--card-bg); }}

      div[data-testid="stChatMessage"] {{ background: transparent !important; border-radius: 12px; padding: 0.4rem 0.2rem; }}

      .source-card {{
        border: 1px solid var(--border); border-radius: 10px; padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem; background: var(--card-bg); color: var(--ink);
      }}
      .confidence-pill {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; background: var(--brand-light);
        color: var(--brand); margin-right: 8px;
      }}
      .system-note {{
        border: 1px dashed var(--border); border-radius: 10px; padding: 0.5rem 0.8rem;
        color: var(--muted); font-size: 0.88rem; background: var(--card-bg);
      }}
      .sidebar-caption {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
        color: var(--muted); margin: 0.6rem 0 0.15rem 0.2rem;
      }}
      [data-testid="stChatInput"] {{ background: var(--card-bg) !important; border-color: var(--border) !important; }}
      [data-testid="stChatInput"] textarea {{ color: var(--ink) !important; }}

      .io-toolbar {{ display: flex; justify-content: flex-end; margin-bottom: 0.15rem; }}
      .st-key-toggle_auto_read button {{
        border-radius: 999px !important; padding: 0.2rem 0.6rem !important;
      }}

      /* Sidebar: conversation list scrolls; Settings stays pinned to the
         bottom of the sidebar regardless of how many chats are listed. */
      [data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; height: 100%; }}
      .st-key-sidebar_scroll {{ flex: 1 1 auto; overflow-y: auto; }}
      .st-key-settings_anchor {{
        position: sticky; bottom: 0; background: var(--sidebar-bg);
        padding-top: 0.5rem; margin-top: auto;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state — current conversation
# ----------------------------------------------------------------------
def _load_conv_into_state(conv: dict) -> None:
    st.session_state.conversation_id = conv["id"]
    st.session_state.conv_title = conv["title"]
    st.session_state.conv_created_at = conv["created_at"]
    st.session_state.namespace = conv["namespace"]
    st.session_state.processed_docs = conv["processed_docs"]
    st.session_state.chat_messages = conv["messages"]


def start_new_chat() -> None:
    _load_conv_into_state(store.new_conversation())


def open_conversation(conv_id: str) -> None:
    conv = store.load_conversation(conv_id)
    if conv:
        _load_conv_into_state(conv)


if "conversation_id" not in st.session_state:
    start_new_chat()
if "auto_read_aloud" not in st.session_state:
    st.session_state.auto_read_aloud = False
if "tts_text" not in st.session_state:
    st.session_state.tts_text = None


def persist_current_conversation() -> None:
    store.save_conversation({
        "id": st.session_state.conversation_id,
        "title": st.session_state.conv_title,
        "created_at": st.session_state.conv_created_at,
        "namespace": st.session_state.namespace,
        "processed_docs": st.session_state.processed_docs,
        "messages": st.session_state.chat_messages,
    })


def add_message(role: str, content: str, meta: dict | None = None) -> None:
    st.session_state.chat_messages.append({"role": role, "content": content, "meta": meta})
    if role == "user" and st.session_state.conv_title == "New chat":
        title = " ".join(content.strip().split())
        st.session_state.conv_title = (title[:40] + "…") if len(title) > 40 else title
    persist_current_conversation()


def get_vector_store() -> PineconeVectorStore | None:
    try:
        return PineconeVectorStore()
    except VectorStoreError as e:
        st.error(f"Pinecone connection failed: {e}", icon=":material/error:")
        return None


def speak(text: str) -> None:
    """Queue text to be read aloud on this rerun via the browser's
    built-in speech synthesis (no external TTS API/key needed)."""
    st.session_state.tts_text = text


def group_label(iso_ts: str) -> str:
    try:
        d = datetime.fromisoformat(iso_ts).date()
    except ValueError:
        return "Older"
    delta = (date.today() - d).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta <= 7:
        return "Previous 7 days"
    return "Older"


# ----------------------------------------------------------------------
# Settings dialog — Appearance + Logs (replaces visible config controls)
# ----------------------------------------------------------------------
@st.dialog("Settings")
def settings_dialog():
    st.subheader("Appearance")
    label_map = {"light": "Light", "dark": "Dark", "system": "System default"}
    reverse_map = {v: k for k, v in label_map.items()}
    choice = st.radio(
        "Theme", list(label_map.values()),
        index=list(label_map.keys()).index(st.session_state.theme),
        horizontal=True, label_visibility="collapsed",
    )
    new_theme = reverse_map[choice]
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        store.save_settings({"theme": new_theme})
        st.rerun()

    st.markdown("---")
    st.subheader("Logs")
    st.caption("Every question asked (across all chats) is logged locally for analysis.")
    logs = read_recent_logs(50)
    if not logs:
        st.caption("No logs yet.")
    else:
        st.dataframe(
            [
                {"timestamp": l["timestamp"], "query": l["query"],
                 "confidence": l.get("confidence"), "latency_ms": l.get("latency_ms"),
                 "namespace": l.get("namespace")}
                for l in reversed(logs)
            ],
            use_container_width=True, height=220,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(log_file_path()):
            with open(log_file_path(), "rb") as f:
                st.download_button("Download logs", f.read(), file_name="lexora_query_log.jsonl",
                                    use_container_width=True, icon=":material/download:")
        else:
            st.button("Download logs", disabled=True, use_container_width=True, icon=":material/download:")
    with col_b:
        if st.button("Clear logs", use_container_width=True, icon=":material/delete:"):
            st.session_state.confirm_clear_logs = True

    if st.session_state.get("confirm_clear_logs"):
        st.warning("This permanently deletes all logged queries. Are you sure?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, clear logs", type="primary", use_container_width=True):
            clear_logs()
            st.session_state.confirm_clear_logs = False
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.confirm_clear_logs = False
            st.rerun()


# ----------------------------------------------------------------------
# Sidebar — brand, new chat, saved conversations, retrieval info, settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Lexora")

    problems = validate_required_keys()
    if problems:
        st.error("Missing configuration:\n- " + "\n- ".join(problems), icon=":material/error:")
        st.caption("Set these in a `.env` file (see README).")

    if st.button("New chat", use_container_width=True, type="primary", icon=":material/add:"):
        start_new_chat()
        st.rerun()

    # Everything below scrolls as one region; Settings (outside this
    # container, further down) stays pinned to the bottom of the sidebar.
    with st.container(key="sidebar_scroll"):
        conversations = store.list_conversations()
        if conversations:
            groups: dict[str, list] = {}
            for c in conversations:
                groups.setdefault(group_label(c["updated_at"]), []).append(c)
            for label in ["Today", "Yesterday", "Previous 7 days", "Older"]:
                if label not in groups:
                    continue
                st.markdown(f"<div class='sidebar-caption'>{label}</div>", unsafe_allow_html=True)
                for c in groups[label]:
                    is_active = c["id"] == st.session_state.conversation_id
                    if st.button(
                        c["title"], key=f"conv_{c['id']}", use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        open_conversation(c["id"])
                        st.rerun()

        st.markdown("---")
        with st.expander("This chat's documents & retrieval", icon=":material/attach_file:"):
            doc_filter = None
            if st.session_state.processed_docs:
                doc_options = ["All documents"] + sorted(set(st.session_state.processed_docs))
                selected_doc = st.selectbox("Restrict answers to", doc_options)
                doc_filter = None if selected_doc == "All documents" else selected_doc
                for d in sorted(set(st.session_state.processed_docs)):
                    st.markdown(f"- {d}")
            else:
                st.caption("Nothing attached yet — use the **+** button in the message box.")

            last_sources = None
            for m in reversed(st.session_state.chat_messages):
                if m["role"] == "assistant" and m.get("meta") and m["meta"].get("sources"):
                    last_sources = m["meta"]["sources"]
                    break
            if last_sources:
                st.caption("Chunks used in the last answer:")
                st.dataframe(
                    [{"document": s["doc_name"], "page": s["page_number"], "similarity": round(s["score"], 2)}
                     for s in last_sources],
                    use_container_width=True, hide_index=True,
                )

            st.markdown("---")
            chunk_size = st.slider("Chunk size (characters)", 200, 2000, settings.default_chunk_size, step=50)
            chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, settings.default_chunk_overlap, step=10)
            top_k = st.slider("Top-K chunks to retrieve", 1, 15, settings.default_top_k)
            score_threshold = st.slider("Similarity threshold (cosine)", 0.0, 1.0, settings.default_score_threshold, step=0.05)
            page_filter_enabled = st.checkbox("Filter by specific page number")
            page_filter_value = st.number_input("Page number", min_value=1, value=1, step=1) if page_filter_enabled else None

            if st.session_state.processed_docs and st.button(
                "Clear this chat's documents", icon=":material/delete:"
            ):
                vs = get_vector_store()
                if vs:
                    try:
                        vs.delete_namespace(st.session_state.namespace)
                        st.session_state.processed_docs = []
                        persist_current_conversation()
                        st.success("Cleared.")
                        st.rerun()
                    except VectorStoreError as e:
                        st.error(str(e))

    with st.container(key="settings_anchor"):
        st.markdown("---")
        if st.button("Settings", use_container_width=True, icon=":material/settings:"):
            settings_dialog()

# ----------------------------------------------------------------------
# Main layout — just the chat
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header"><h1>Lexora</h1></div>
    <div class="app-subtitle">
      Attach PDFs — even scanned ones, OCR kicks in automatically — right from the
      message box, then ask questions by typing or speaking. Answers come strictly
      from your documents.
    </div>
    """,
    unsafe_allow_html=True,
)


def index_one_file(uploaded_file, vs, embedder) -> str:
    """Extracts, chunks, embeds and upserts a single uploaded file.
    Returns a short human-readable status line. Raises nothing — all
    errors are caught and returned as the status line so one bad file
    doesn't stop the rest of the batch."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        pages = extract_text_from_pdf(tmp_path, doc_name=uploaded_file.name)
        if not any(p.text.strip() for p in pages):
            return f"**{uploaded_file.name}** — no extractable text (even after OCR); skipped."

        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts).tolist()
        metadatas = [
            {"text": c.text, "doc_name": c.doc_name, "page_number": c.page_number,
             "chunk_id": c.chunk_id, "used_ocr": c.used_ocr}
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]
        vs.upsert_chunks(ids=ids, vectors=vectors, metadatas=metadatas, namespace=st.session_state.namespace)

        st.session_state.processed_docs.append(uploaded_file.name)
        ocr_pages = sum(1 for p in pages if p.used_ocr)
        return f"**{uploaded_file.name}** — {len(pages)} pages, {len(chunks)} chunks ({ocr_pages} used OCR)."
    except PDFExtractionError as e:
        return f"**{uploaded_file.name}** — invalid or unreadable PDF: {e}"
    except VectorStoreError as e:
        return f"**{uploaded_file.name}** — Pinecone error: {e}"
    except Exception as e:
        return f"**{uploaded_file.name}** — unexpected error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def index_files(files) -> None:
    """Auto-indexes one or more attached files — this is the whole
    'upload and index' step now: there is no separate index button,
    it just happens the moment files are attached to a message."""
    vs = get_vector_store()
    if vs is None:
        return
    embedder = get_embedder()
    lines = []
    with st.spinner(f"Indexing {len(files)} document(s)..."):
        for f in files:
            lines.append(index_one_file(f, vs, embedder))
    add_message("system", "\n".join(lines))


def run_query(query_text: str, doc_filter) -> None:
    """Runs retrieve -> generate for one question, appends the turn
    (both user question and assistant answer) to the chat transcript."""
    add_message("user", query_text)

    if not st.session_state.processed_docs:
        add_message("assistant", "No documents have been attached to this chat yet — "
                                  "use the **+** button in the message box first.")
        return

    vs = get_vector_store()
    if vs is None:
        return
    try:
        start = time.time()
        embedder = get_embedder()
        retriever = Retriever(vector_store=vs, embedder=embedder)
        chunks = retriever.retrieve(
            query=query_text, top_k=top_k, namespace=st.session_state.namespace,
            score_threshold=score_threshold, page_filter=page_filter_value, doc_filter=doc_filter,
        )
        generator = GroqGenerator()
        result = generator.generate(query_text, chunks)
        confidence = estimate_confidence(chunks, result)
        latency_ms = round((time.time() - start) * 1000, 1)

        sources = [{"doc_name": c.doc_name, "page_number": c.page_number, "score": c.score, "text": c.text}
                   for c in chunks]
        log_query(
            query=query_text, answer=result.answer, namespace=st.session_state.namespace, top_k=top_k,
            score_threshold=score_threshold,
            sources=[{"doc_name": s["doc_name"], "page_number": s["page_number"], "score": s["score"]} for s in sources],
            confidence=confidence, latency_ms=latency_ms,
        )
        add_message("assistant", result.answer, meta={
            "confidence": confidence, "latency_ms": latency_ms, "sources": sources,
        })
        if st.session_state.auto_read_aloud:
            speak(result.answer)
    except ValueError as e:
        add_message("assistant", f"{e}")
    except VectorStoreError as e:
        add_message("assistant", f"Pinecone error: {e}")
    except RuntimeError as e:
        add_message("assistant", f"Groq LLM error: {e}")
    except Exception as e:
        add_message("assistant", f"Unexpected error: {e}")


# ---- Render the running transcript -------------------------------------
for i, msg in enumerate(st.session_state.chat_messages):
    if msg["role"] == "system":
        with st.chat_message("assistant", avatar=":material/attach_file:"):
            st.markdown(f"<div class='system-note'>{msg['content']}</div>", unsafe_allow_html=True)
        continue

    avatar = ":material/person:" if msg["role"] == "user" else ":material/auto_awesome:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        meta = msg.get("meta")
        if meta:
            st.markdown(
                f"<span class='confidence-pill'>Confidence {meta['confidence']}%</span>"
                f"<span class='confidence-pill'>{meta['latency_ms']} ms</span>",
                unsafe_allow_html=True,
            )
            if meta.get("sources"):
                with st.expander(f"{len(meta['sources'])} source(s)", icon=":material/description:"):
                    for j, s in enumerate(meta["sources"], start=1):
                        st.markdown(
                            f"<div class='source-card'><b>Source {j}</b> — {s['doc_name']}, "
                            f"page {s['page_number']} (similarity: {s['score']:.2f})<br>{s['text']}</div>",
                            unsafe_allow_html=True,
                        )

# ---- Fire off any queued text-to-speech (browser-native, no API key) ---
if st.session_state.tts_text:
    text_to_speak = st.session_state.tts_text
    st.session_state.tts_text = None
    components.html(
        f"""
        <script>
          try {{
            const u = new SpeechSynthesisUtterance({json.dumps(text_to_speak)});
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
          }} catch (e) {{ console.error("TTS failed", e); }}
        </script>
        """,
        height=0,
    )

# ---- Slim toolbar fused to the top of the message box: voice-output ----
st.markdown("<div class='io-toolbar'>", unsafe_allow_html=True)
icon = ":material/volume_up:" if st.session_state.auto_read_aloud else ":material/volume_off:"
if st.button(icon, key="toggle_auto_read", help="Read answers aloud automatically"):
    st.session_state.auto_read_aloud = not st.session_state.auto_read_aloud
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ---- The single ChatGPT-style input: text + "+" attach + mic + send ----
prompt = st.chat_input(
    "Message Lexora — ask a question or attach PDFs...",
    accept_file="multiple",
    file_type=["pdf"],
    accept_audio=True,
)

if prompt:
    files = prompt.files if prompt.files else []
    audio = prompt.audio
    text = (prompt.text or "").strip()

    oversized = [f.name for f in files if f.size > settings.max_upload_mb * 1024 * 1024]
    files = [f for f in files if f.name not in oversized]
    if oversized:
        add_message("system", f"Skipped (over {settings.max_upload_mb} MB): {', '.join(oversized)}")

    if files:
        index_files(files)

    if not text and audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio.read())
            tmp_audio_path = tmp_audio.name
        try:
            with st.spinner("Transcribing..."):
                text = transcribe_audio(tmp_audio_path)
            if not text:
                add_message("system", "No speech detected in the recording — try again.")
        except STTError as e:
            add_message("system", f"{e}")
        except Exception as e:
            # Catch anything transcribe_audio didn't wrap (e.g. a missing
            # model download or codec issue) so it's never a silent failure.
            add_message("system", f"Speech-to-text failed unexpectedly: {e}")
        finally:
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)

    if text:
        run_query(text, doc_filter)

    st.rerun()
