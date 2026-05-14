import streamlit as st
import time
import re
import zipfile
import io
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE_URL = "https://platform.beeknoee.com/api/v1"

SYSTEM_PROMPT = """Bạn là dịch giả tiểu thuyết tiên hiệp Trung Hoa sang tiếng Việt hàng đầu. Yêu cầu BẮT BUỘC:
- Chỉ trả về bản dịch tiếng Việt.
- Không được giải thích, không suy nghĩ thành tiếng, không ghi chú, không phân tích.
- Không được viết các cụm như: "Để tôi dịch", "Bản dịch", "Thinking", "Phân tích".
- Dịch đầy đủ từng câu, tuyệt đối không tóm tắt.
- Giữ văn phong tiên hiệp cổ phong tự nhiên.
- Giữ nguyên tên riêng, công pháp, cảnh giới theo Hán Việt.
- Không dùng từ hiện đại, tiếng lóng."""

SYSTEM_PROMPT_FIX = """Bạn là dịch giả tiểu thuyết tiên hiệp Trung Hoa sang tiếng Việt hàng đầu.
Đoạn sau đây vẫn còn chữ Hán chưa được dịch. Hãy dịch TOÀN BỘ sang tiếng Việt.
Yêu cầu BẮT BUỘC:
- Chỉ trả về bản dịch tiếng Việt, không kèm chú thích hay giải thích.
- Dịch đầy đủ từng câu, tuyệt đối không tóm tắt.
- Giữ văn phong tiên hiệp cổ phong tự nhiên.
- Giữ nguyên tên riêng, công pháp, cảnh giới theo Hán Việt.
- Không dùng từ hiện đại, tiếng lóng."""

CJK_RE = re.compile(
    r'[\u4e00-\u9fff'
    r'\u3400-\u4dbf'
    r'\uf900-\ufaff'
    r'\u2e80-\u2eff'
    r'\u31c0-\u31ef'
    r'\U00020000-\U0002a6df'
    r']+'
)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class FileResult:
    name: str
    source: str
    translated: str = ""
    fixed: str = ""
    chunks_total: int = 0
    chunks_done: int = 0
    errors: list = field(default_factory=list)
    han_found: int = 0
    han_fixed: int = 0
    status: str = "pending"

    @property
    def final_text(self):
        return self.fixed if self.fixed else self.translated


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int) -> list:
    paragraphs = text.splitlines()
    chunks, current, size = [], [], 0
    for line in paragraphs:
        if size + len(line) + 1 > chunk_size and current:
            chunks.append("\n".join(current))
            current, size = [line], len(line)
        else:
            current.append(line)
            size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


def call_api(client: OpenAI, model: str, text: str, system: str = SYSTEM_PROMPT) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=8096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": text},
        ],
    )
    return resp.choices[0].message.content


def api_with_retry(client: OpenAI, model: str, text: str,
                   system: str = SYSTEM_PROMPT,
                   status_ph=None) -> tuple:
    for attempt in range(3):
        try:
            return call_api(client, model, text, system), None
        except Exception as e:
            msg = str(e)
            if "rate" in msg.lower() or "429" in msg:
                wait = 30 * (attempt + 1)
                if status_ph:
                    status_ph.warning(f"⚠️ Rate limit — chờ {wait}s…")
                time.sleep(wait)
            else:
                return "", msg
    return "", "Lỗi sau 3 lần thử"


def find_han_segments(text: str) -> list:
    result = []
    for i, line in enumerate(text.split("\n")):
        if CJK_RE.search(line) and line.strip():
            result.append((i, line))
    return result


def build_zip(results: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r.status == "done":
                stem = r.name.rsplit(".", 1)[0]
                zf.writestr(f"{stem}_vi.txt", r.final_text.encode("utf-8"))
    buf.seek(0)
    return buf.read()


def fetch_models(api_key: str) -> list:
    """Fetch available models from the API."""
    try:
        client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
        models = client.models.list()
        ids = sorted([m.id for m in models.data])
        return ids
    except Exception as e:
        return []


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dịch Thuật Tiên Hiệp", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
:root { --red:#c0392b; --dark:#1e1e2e; --text:#cdd6f4; --border:#313244; }
.main-title { text-align:center; font-size:2.2rem; font-weight:800; color:var(--red); margin-bottom:0; }
.sub-title   { text-align:center; color:#7f8c8d; font-style:italic; margin-top:0; margin-bottom:1.2rem; }
.result-box  {
    background:var(--dark); color:var(--text);
    padding:1rem 1.2rem; border-radius:10px;
    font-size:.92rem; line-height:1.75;
    white-space:pre-wrap; max-height:420px; overflow-y:auto;
    border:1px solid var(--border);
}
.stat-row { display:flex; gap:1rem; margin:.5rem 0; flex-wrap:wrap; }
.stat-card {
    background:#f8f9fa; border-left:4px solid var(--red);
    padding:.5rem 1rem; border-radius:4px;
    font-size:.83rem; flex:1; min-width:140px;
}
.api-info {
    background:#eaf4fb; border-left:4px solid #2980b9;
    padding:.5rem 1rem; border-radius:4px;
    font-size:.82rem; margin-bottom:.8rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚔️ Dịch Thuật Tiên Hiệp</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Trung Hoa → Tiếng Việt · Văn phong cổ phong · Tự động kiểm tra & vá chữ Hán sót</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    st.markdown(
        '<div class="api-info">🔑 API: <b>platform.beeknoee.com</b></div>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Nhập key từ platform.beeknoee.com…",
    )

    # Model selector — fetch list or manual input
    model_input_mode = st.radio("Chọn model", ["Tự động lấy danh sách", "Nhập tay"], horizontal=True)

    model = ""
    if model_input_mode == "Tự động lấy danh sách":
        fetch_btn = st.button("🔄 Lấy danh sách model", use_container_width=True)
        if fetch_btn:
            if not api_key:
                st.error("Nhập API key trước.")
            else:
                with st.spinner("Đang lấy danh sách…"):
                    model_list = fetch_models(api_key)
                if model_list:
                    st.session_state["model_list"] = model_list
                    st.success(f"✅ Tìm thấy {len(model_list)} model")
                else:
                    st.warning("Không lấy được danh sách. Hãy nhập tay.")

        if "model_list" in st.session_state and st.session_state["model_list"]:
            model = st.selectbox("Model", st.session_state["model_list"])
        else:
            st.caption("Nhấn nút trên để tải danh sách model.")
    else:
        model = st.text_input("Tên model", placeholder="vd: gpt-4o, claude-3-5-sonnet…")

    st.divider()
    chunk_size = st.slider("Chunk size (ký tự)", 500, 6000, 2000, 100)
    delay      = st.slider("Delay giữa chunk (giây)", 0.0, 5.0, 0.3, 0.1)
    do_check   = st.toggle("🔍 Kiểm tra & vá chữ Hán sót", value=True)

    st.divider()
    st.markdown("""**Quy trình:**
1. Nhập API key → lấy danh sách model
2. Upload nhiều file `.txt`
3. Nhấn **Dịch Thuật**
4. Tự kiểm tra & vá chữ Hán sót từng dòng
5. Tải từng file hoặc **tất cả (ZIP)**""")


# ── Input tabs ────────────────────────────────────────────────────────────────
tab_file, tab_text = st.tabs(["📁 Upload nhiều File .txt", "✏️ Dán Văn Bản"])
uploaded_list = []

with tab_file:
    files = st.file_uploader(
        "Chọn một hoặc nhiều file .txt tiếng Trung",
        type=["txt"],
        accept_multiple_files=True,
        help="Hỗ trợ UTF-8 / GBK / GB2312 / Big5",
    )
    if files:
        for f in files:
            raw = f.read()
            decoded = None
            for enc in ("utf-8", "gb18030", "gbk", "big5"):
                try:
                    decoded = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded:
                uploaded_list.append((f.name, decoded))
            else:
                st.error(f"❌ Không đọc được `{f.name}` — hãy chuyển sang UTF-8.")
        if uploaded_list:
            st.success(f"✅ Đọc thành công **{len(uploaded_list)}** file")
            with st.expander("Xem danh sách"):
                for name, txt in uploaded_list:
                    st.markdown(f"- **{name}** — {len(txt):,} ký tự")

with tab_text:
    pasted = st.text_area("Dán văn bản tiếng Trung:", height=220, placeholder="粘贴中文原文到这里…")
    if pasted.strip():
        uploaded_list = [("van_ban_dan.txt", pasted.strip())]


# ── Translate ─────────────────────────────────────────────────────────────────
st.divider()
translate_btn = st.button("🔄 Bắt Đầu Dịch Thuật", type="primary", use_container_width=True)

if translate_btn:
    if not api_key:
        st.error("⛔ Vui lòng nhập **API Key** ở sidebar.")
        st.stop()
    if not model:
        st.error("⛔ Vui lòng chọn hoặc nhập **tên model**.")
        st.stop()
    if not uploaded_list:
        st.error("⛔ Chưa có file / văn bản để dịch.")
        st.stop()

    client  = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    results = [FileResult(name=n, source=t) for n, t in uploaded_list]

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">📂 Số file: <b>{len(results)}</b></div>
      <div class="stat-card">🤖 Model: <b>{model}</b></div>
      <div class="stat-card">📏 Chunk: <b>{chunk_size:,} ký tự</b></div>
      <div class="stat-card">🔍 Kiểm tra Hán: <b>{"Bật" if do_check else "Tắt"}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Build per-file UI containers
    file_containers = []
    for r in results:
        with st.expander(f"📄 {r.name}", expanded=True):
            c_status   = st.empty()
            c_progress = st.empty()
            c_result   = st.empty()
            c_check    = st.empty()
            c_dl       = st.empty()
        file_containers.append((c_status, c_progress, c_result, c_check, c_dl))

    # ── Translate each file ───────────────────────────────────────────────────
    for idx, r in enumerate(results):
        c_status, c_progress, c_result, c_check, c_dl = file_containers[idx]
        r.status = "translating"
        chunks = split_text(r.source, chunk_size)
        r.chunks_total = len(chunks)
        parts  = []

        c_status.info(f"⏳ **{r.name}** — dịch {len(chunks)} chunk…")

        for i, chunk in enumerate(chunks):
            c_progress.progress(i / len(chunks), text=f"Chunk {i+1}/{len(chunks)}")
            translated, err = api_with_retry(client, model, chunk, status_ph=c_status)
            parts.append(translated if not err else f"[LỖI CHUNK {i+1}: {err}]")
            if err:
                r.errors.append(f"Chunk {i+1}: {err}")
            r.chunks_done = i + 1

            live = "\n\n".join(parts)
            c_result.markdown(
                f'<div class="result-box">{live.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            if delay > 0 and i < len(chunks) - 1:
                time.sleep(delay)

        c_progress.progress(1.0, text="✅ Dịch xong")
        r.translated = "\n\n".join(parts)

        # ── Check & fix leftover Han ──────────────────────────────────────────
        if do_check:
            r.status = "checking"
            han_segs = find_han_segments(r.translated)
            r.han_found = len(han_segs)

            if not han_segs:
                c_check.success(f"✅ Không còn chữ Hán sót trong **{r.name}**.")
                r.fixed = r.translated
            else:
                c_check.warning(f"⚠️ Phát hiện **{r.han_found}** dòng còn chữ Hán — đang vá…")
                r.status = "fixing"
                lines = r.translated.split("\n")
                fix_ph = st.empty()

                for fi, (li, bad_line) in enumerate(han_segs):
                    fix_ph.caption(
                        f"🔧 Vá dòng {fi+1}/{r.han_found}: `{bad_line[:60]}{'…' if len(bad_line)>60 else ''}`"
                    )
                    fixed_line, err = api_with_retry(
                        client, model, bad_line,
                        system=SYSTEM_PROMPT_FIX,
                        status_ph=c_check,
                    )
                    if not err and fixed_line.strip():
                        lines[li] = fixed_line
                        r.han_fixed += 1
                    if delay > 0:
                        time.sleep(delay)

                fix_ph.empty()
                r.fixed = "\n".join(lines)

                remaining = find_han_segments(r.fixed)
                if remaining:
                    c_check.warning(
                        f"⚠️ Đã vá **{r.han_fixed}/{r.han_found}** dòng. "
                        f"Còn **{len(remaining)}** dòng (có thể là tên riêng/tiêu đề)."
                    )
                else:
                    c_check.success(f"✅ Đã vá toàn bộ **{r.han_fixed}** dòng sót!")
        else:
            r.fixed = r.translated

        r.status = "done"
        c_status.success(f"🎉 **{r.name}** — hoàn tất!")
        stem = r.name.rsplit(".", 1)[0]
        c_dl.download_button(
            label=f"⬇️ Tải {stem}_vi.txt",
            data=r.final_text.encode("utf-8"),
            file_name=f"{stem}_vi.txt",
            mime="text/plain",
            key=f"dl_{idx}",
        )

    # ── Download ALL zip ──────────────────────────────────────────────────────
    done = [r for r in results if r.status == "done"]
    if len(done) > 1:
        st.divider()
        st.download_button(
            label=f"📦 Tải tất cả {len(done)} file (ZIP)",
            data=build_zip(done),
            file_name="ban_dich_tien_hiep.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Tổng kết")
    cols = st.columns(max(len(results), 1))
    for col, r in zip(cols, results):
        with col:
            icon = "🟢" if r.status == "done" else "🔴"
            st.markdown(f"**{icon} {r.name}**")
            st.markdown(f"- Chunks: {r.chunks_done}/{r.chunks_total}")
            if do_check:
                st.markdown(f"- Hán sót: {r.han_found} dòng")
                st.markdown(f"- Đã vá: {r.han_fixed} dòng")
            if r.errors:
                with st.expander(f"⚠️ {len(r.errors)} lỗi"):
                    for e in r.errors:
                      st.text(e)
