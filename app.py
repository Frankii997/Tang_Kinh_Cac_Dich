import streamlit as st
import time
import re
import zipfile
import io
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

# ── Constants ─────────────────────────────────
PRIMARY_BASE_URL = "https://platform.beeknoee.com/api/v1"

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
    r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'
    r'\u2e80-\u2eff\u31c0-\u31ef\U00020000-\U0002a6df]+'
)

@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    enabled: bool = True

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

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

def call_with_fallback(providers: list, text: str, system: str = SYSTEM_PROMPT, status_ph=None) -> tuple:
    last_error = None
    for prov in providers:
        if not prov.enabled:
            continue
        for attempt in range(2):
            try:
                result = call_api(prov.client(), prov.model, text, system)
                return result, prov.name, None
            except Exception as e:
                msg = str(e)
                if ("rate" in msg.lower() or "429" in msg) and attempt == 0:
                    wait = 20
                    if status_ph:
                        status_ph.warning(f"⚠️ [{prov.name}] Rate limit — chờ {wait}s…")
                    time.sleep(wait)
                else:
                    last_error = f"[{prov.name}] {msg}"
                    if status_ph:
                        status_ph.warning(f"⚠️ {last_error} → thử provider tiếp theo…")
                    break
    return "", None, last_error

def fetch_models(api_key: str, base_url: str) -> list:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception:
        return []

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

st.set_page_config(page_title="Dịch Thuật Tiên Hiệp", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
:root{--red:#c0392b;--dark:#1e1e2e;--text:#cdd6f4;--border:#313244}
.main-title{text-align:center;font-size:2.2rem;font-weight:800;color:var(--red);margin-bottom:0}
.sub-title{text-align:center;color:#7f8c8d;font-style:italic;margin-top:0;margin-bottom:1.2rem}
.result-box{background:var(--dark);color:var(--text);padding:1rem 1.2rem;border-radius:10px;
  font-size:.92rem;line-height:1.75;white-space:pre-wrap;max-height:420px;overflow-y:auto;
  border:1px solid var(--border)}
.stat-row{display:flex;gap:1rem;margin:.5rem 0;flex-wrap:wrap}
.stat-card{background:#f8f9fa;border-left:4px solid var(--red);padding:.5rem 1rem;
  border-radius:4px;font-size:.83rem;flex:1;min-width:140px}
.provider-primary{background:#eaf4fb;border-left:4px solid #2980b9;
  padding:.5rem 1rem;border-radius:4px;font-size:.82rem;margin-bottom:.5rem}
.provider-fallback{background:#fef9e7;border-left:4px solid #f39c12;
  padding:.5rem 1rem;border-radius:4px;font-size:.82rem;margin-bottom:.5rem}
.badge-used{display:inline-block;padding:.15rem .5rem;border-radius:9999px;
  font-size:.72rem;font-weight:700;margin-left:.4rem;vertical-align:middle}
.badge-primary{background:#2980b9;color:#fff}
.badge-fallback{background:#e67e22;color:#fff}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚔️ Dịch Thuật Tiên Hiệp</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Trung Hoa → Tiếng Việt · Tự động fallback khi API lỗi · Kiểm tra & vá chữ Hán sót</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Cài đặt")

    st.markdown('<div class="provider-primary">🔵 <b>Provider chính</b> — platform.beeknoee.com</div>', unsafe_allow_html=True)
    p_key = st.text_input("API Key (beeknoee)", type="password", placeholder="sk-…", key="p_key")
    p_model_mode = st.radio("Model (chính)", ["Tự động lấy", "Nhập tay"], horizontal=True, key="p_mode")
    p_model = ""
    if p_model_mode == "Tự động lấy":
        if st.button("🔄 Lấy model", key="p_fetch", use_container_width=True):
            if p_key:
                with st.spinner("Đang lấy…"):
                    ml = fetch_models(p_key, PRIMARY_BASE_URL)
                if ml:
                    st.session_state["p_model_list"] = ml
                    st.success(f"✅ {len(ml)} model")
                else:
                    st.warning("Không lấy được. Nhập tay.")
        if st.session_state.get("p_model_list"):
            p_model = st.selectbox("", st.session_state["p_model_list"], key="p_sel")
        else:
            st.caption("Nhấn nút trên để tải danh sách.")
    else:
        p_model = st.text_input("Tên model", placeholder="vd: gpt-4o", key="p_manual")

    st.divider()

    st.markdown('<div class="provider-fallback">🟡 <b>Fallback</b> — DS2API (DeepSeek)</div>', unsafe_allow_html=True)
    fb_enabled = st.toggle("Bật fallback DS2API", value=False, key="fb_on")

    fb_url = fb_key = fb_model = ""
    if fb_enabled:
        fb_url = st.text_input("DS2API URL", placeholder="https://your-ds2api.vercel.app/v1", key="fb_url")
        fb_key = st.text_input("DS2API Key", type="password", placeholder="sk-mykey123", key="fb_key")
        fb_model_mode = st.radio("Model (fallback)", ["Tự động lấy", "Nhập tay"], horizontal=True, key="fb_mode")
        if fb_model_mode == "Tự động lấy":
            if st.button("🔄 Lấy model", key="fb_fetch", use_container_width=True):
                if fb_key and fb_url:
                    with st.spinner("Đang lấy…"):
                        ml = fetch_models(fb_key, fb_url)
                    if ml:
                        st.session_state["fb_model_list"] = ml
                        st.success(f"✅ {len(ml)} model")
                    else:
                        st.warning("Không lấy được. Nhập tay.")
            if st.session_state.get("fb_model_list"):
                fb_model = st.selectbox("", st.session_state["fb_model_list"], key="fb_sel")
            else:
                st.caption("Nhấn nút trên để tải danh sách.")
        else:
            fb_model = st.text_input("Tên model", placeholder="deepseek-v4-flash-nothinking", key="fb_manual")

    st.divider()
    chunk_size = st.slider("Chunk size (ký tự)", 500, 6000, 2000, 100)
    delay      = st.slider("Delay giữa chunk (giây)", 0.0, 5.0, 0.3, 0.1)
    do_check   = st.toggle("🔍 Kiểm tra & vá chữ Hán sót", value=True)

tab_file, tab_text = st.tabs(["📁 Upload nhiều File .txt", "✏️ Dán Văn Bản"])
uploaded_list = []

with tab_file:
    files = st.file_uploader("Chọn một hoặc nhiều file .txt tiếng Trung",
        type=["txt"], accept_multiple_files=True)
    if files:
        for f in files:
            raw = f.read()
            decoded = None
            for enc in ("utf-8", "gb18030", "gbk", "big5"):
                try:
                    decoded = raw.decode(enc); break
                except UnicodeDecodeError:
                    continue
            if decoded:
                uploaded_list.append((f.name, decoded))
            else:
                st.error(f"❌ Không đọc được `{f.name}`")
        if uploaded_list:
            st.success(f"✅ {len(uploaded_list)} file")
            with st.expander("Danh sách file"):
                for name, txt in uploaded_list:
                    st.markdown(f"- **{name}** — {len(txt):,} ký tự")

with tab_text:
    pasted = st.text_area("Dán văn bản tiếng Trung:", height=220, placeholder="粘贴中文原文到这里…")
    if pasted.strip():
        uploaded_list = [("van_ban_dan.txt", pasted.strip())]

st.divider()
translate_btn = st.button("🔄 Bắt Đầu Dịch Thuật", type="primary", use_container_width=True)

if translate_btn:
    providers = []
    if p_key and p_model:
        providers.append(Provider("beeknoee", PRIMARY_BASE_URL, p_key, p_model))
    if fb_enabled and fb_url and fb_key and fb_model:
        providers.append(Provider("ds2api", fb_url, fb_key, fb_model))

    if not providers:
        st.error("⛔ Chưa cấu hình provider nào. Nhập API key + model ở sidebar.")
        st.stop()
    if not uploaded_list:
        st.error("⛔ Chưa có file / văn bản để dịch.")
        st.stop()

    results = [FileResult(name=n, source=t) for n, t in uploaded_list]
    provider_usage = {p.name: 0 for p in providers}

    st.markdown(f"""<div class="stat-row">
      <div class="stat-card">📂 File: <b>{len(results)}</b></div>
      <div class="stat-card">🔗 Providers: <b>{" → ".join(p.name for p in providers)}</b></div>
      <div class="stat-card">📏 Chunk: <b>{chunk_size:,} ký tự</b></div>
      <div class="stat-card">🔍 Hán check: <b>{"Bật" if do_check else "Tắt"}</b></div>
    </div>""", unsafe_allow_html=True)

    file_containers = []
    for r in results:
        with st.expander(f"📄 {r.name}", expanded=True):
            file_containers.append((st.empty(), st.empty(), st.empty(), st.empty(), st.empty()))

    for idx, r in enumerate(results):
        c_status, c_progress, c_result, c_check, c_dl = file_containers[idx]
        chunks = split_text(r.source, chunk_size)
        r.chunks_total = len(chunks)
        parts = []
        c_status.info(f"⏳ **{r.name}** — {len(chunks)} chunk…")

        for i, chunk in enumerate(chunks):
            c_progress.progress(i / len(chunks), text=f"Chunk {i+1}/{len(chunks)}")
            translated, used, err = call_with_fallback(providers, chunk, status_ph=c_status)
            if err:
                r.errors.append(f"Chunk {i+1}: {err}")
                parts.append(f"[LỖI CHUNK {i+1}: {err}]")
                badge = ""
            else:
                parts.append(translated)
                if used:
                    provider_usage[used] = provider_usage.get(used, 0) + 1
                badge_cls = "badge-primary" if used == "beeknoee" else "badge-fallback"
                badge = f'<span class="badge-used {badge_cls}">{used}</span>'

            live = "\n\n".join(parts)
            c_result.markdown(
                f'<div class="result-box">Chunk {i+1}/{len(chunks)} {badge}<br><br>'
                f'{live.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            r.chunks_done = i + 1
            if delay > 0 and i < len(chunks) - 1:
                time.sleep(delay)

        c_progress.progress(1.0, text="✅ Dịch xong")
        r.translated = "\n\n".join(parts)

        if do_check:
            han_segs = find_han_segments(r.translated)
            r.han_found = len(han_segs)
            if not han_segs:
                c_check.success("✅ Không còn chữ Hán sót.")
                r.fixed = r.translated
            else:
                c_check.warning(f"⚠️ {r.han_found} dòng còn chữ Hán — đang vá…")
                lines = r.translated.split("\n")
                fix_ph = st.empty()
                for fi, (li, bad_line) in enumerate(han_segs):
                    fix_ph.caption(f"🔧 Vá {fi+1}/{r.han_found}: `{bad_line[:60]}{'…' if len(bad_line)>60 else ''}`")
                    fixed_line, used, err = call_with_fallback(providers, bad_line, SYSTEM_PROMPT_FIX, c_check)
                    if not err and fixed_line.strip():
                        lines[li] = fixed_line
                        r.han_fixed += 1
                    if delay > 0:
                        time.sleep(delay)
                fix_ph.empty()
                r.fixed = "\n".join(lines)
                remaining = find_han_segments(r.fixed)
                if remaining:
                    c_check.warning(f"⚠️ Đã vá {r.han_fixed}/{r.han_found}. Còn {len(remaining)} dòng (tên riêng).")
                else:
                    c_check.success(f"✅ Vá xong {r.han_fixed} dòng!")
        else:
            r.fixed = r.translated

        r.status = "done"
        usage_str = " | ".join(f"{n}: {c} chunk" for n, c in provider_usage.items() if c > 0)
        c_status.success(f"🎉 **{r.name}** hoàn tất! ({usage_str})")
        stem = r.name.rsplit(".", 1)[0]
        c_dl.download_button(f"⬇️ Tải {stem}_vi.txt", r.final_text.encode("utf-8"),
            f"{stem}_vi.txt", "text/plain", key=f"dl_{idx}")

    done = [r for r in results if r.status == "done"]
    if len(done) > 1:
        st.divider()
        st.download_button(f"📦 Tải tất cả {len(done)} file (ZIP)", build_zip(done),
            "ban_dich_tien_hiep.zip", "application/zip", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 📊 Tổng kết")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng file", len(results))
    c2.metric("Tổng chunk", sum(r.chunks_total for r in results))
    c3.metric("beeknoee", provider_usage.get("beeknoee", 0))
    c4.metric("ds2api", provider_usage.get("ds2api", 0))
    for r in results:
        with st.expander(f"{'🟢' if r.status=='done' else '🔴'} {r.name}"):
            st.markdown(f"- Chunks: {r.chunks_done}/{r.chunks_total}")
            if do_check:
                st.markdown(f"- Hán sót: {r.han_found} | Đã vá: {r.han_fixed}")
            if r.errors:
                for e in r.errors:
                    st.text(e)
