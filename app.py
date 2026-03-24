import os
import sys
import tempfile
from pathlib import Path

# Windows環境でのASCIIエンコードエラー対策
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# .env を自動検索して読み込む（ローカル開発用）
# 探す順序: プロジェクトルート → 親ディレクトリ → ホームディレクトリ
for _env_candidate in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path.home() / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break

import streamlit as st
from google import genai
import fitz  # PyMuPDF
import docx

# ──────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────
st.set_page_config(
    page_title="RuleAdapt - 就業規則アジャストツール",
    page_icon="⚖",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# ──────────────────────────────────────────
# 知識ベース（起動時に一度だけ読み込む）
# ──────────────────────────────────────────
@st.cache_resource
def load_knowledge_base() -> str:
    kb_file = KNOWLEDGE_DIR / "career_up_qa.txt"
    if kb_file.exists():
        return kb_file.read_text(encoding="utf-8")
    return ""

KNOWLEDGE_BASE = load_knowledge_base()

# ──────────────────────────────────────────
# APIキー取得（st.secrets → 環境変数の順に探す）
# ──────────────────────────────────────────
def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

# ──────────────────────────────────────────
# ファイルからテキスト抽出
# ──────────────────────────────────────────
def extract_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_from_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def extract_text(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if ext in (".docx", ".doc"):
            return extract_from_docx(tmp_path)
        elif ext == ".pdf":
            return extract_from_pdf(tmp_path)
        elif ext in (".txt",):
            return extract_from_txt(tmp_path)
        else:
            raise ValueError(f"対応していないファイル形式です: {ext}")
    finally:
        os.unlink(tmp_path)

# ──────────────────────────────────────────
# プロンプト生成
# ──────────────────────────────────────────
def build_prompt(rule_text: str, reality_memo: str) -> str:
    kb_excerpt = KNOWLEDGE_BASE[:10000]
    return f"""あなたは社会保険労務士の専門アシスタントです。
キャリアアップ助成金（正社員化コース）の支給要領に精通しており、就業規則の問題点を発見・修正提案する専門家です。

## 参考知識：キャリアアップ助成金Q&A（抜粋）
{kb_excerpt}

---

## 分析対象：就業規則テキスト
{rule_text[:8000]}

---

## 実態メモ（現場情報）
{reality_memo if reality_memo.strip() else "（実態メモなし）"}

---

## 指示
上記の就業規則と実態メモを分析し、以下の3つのセクションで回答してください。

### 【① 条文構造チェック】
就業規則の以下のカテゴリについて、記載の有無・問題点・条文間の矛盾を指摘してください。
- 労働時間（週40時間・変形労働時間制等）
- 休日（週1日・4週4日変形等）
- 賃金・手当（台帳との整合性）
- 昇給規定（3%以上の根拠）
- 正規・非正規の待遇差の根拠

### 【② 支給要領チェック＆目検アラート】
キャリアアップ助成金の支給要件と照らし合わせ、以下の3段階で判定してください：
- 🔴「このままでは不支給リスクあり」
- 🟡「条文修正で対応可能」
- 🟢「問題なし」

また、FAX・手書き書類等のアナログ媒体で**必ず目視確認すべきポイント**を具体的に列挙してください。

### 【③ 条文アジャスト提案（処方箋）】
問題のある箇所について、実態を変えずに要件を満たす条文修正案を具体的に提示してください。
- 修正が必要な箇所の現状と問題点
- 修正案（具体的な条文テキスト）
- 注意事項（不利益変更の場合は必ず明記）

回答は日本語で、実務担当者が即座に活用できる具体的な内容にしてください。
"""

# ──────────────────────────────────────────
# UI
# ──────────────────────────────────────────
st.title("⚖ RuleAdapt")
st.caption("助成金対応型・就業規則自動アジャストツール｜キャリアアップ助成金（正社員化コース）")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 就業規則・賃金規定")
    uploaded_file = st.file_uploader(
        "ファイルをアップロード",
        type=["docx", "doc", "pdf", "txt"],
        help="Word(.docx) / PDF / テキストファイルに対応しています",
    )
    rule_text_direct = st.text_area(
        "またはテキストを直接入力",
        height=200,
        placeholder="就業規則の本文をここに貼り付けることもできます...",
    )

with col2:
    st.subheader("📝 実態メモ")
    reality_memo = st.text_area(
        "現場の実態情報を入力してください",
        height=300,
        placeholder=(
            "例：\n"
            "・毎日15分早出して掃除している\n"
            "・残業代が基本給に込みになっている\n"
            "・シフト表では土曜出勤が月2回ある\n"
            "・昇給は口頭で約束しているが規定なし\n"
            "・パートは交通費なし、正社員はあり"
        ),
    )

st.divider()
analyze_btn = st.button("🔍 就業規則を解析する", type="primary", use_container_width=True)

# ──────────────────────────────────────────
# 解析実行
# ──────────────────────────────────────────
if analyze_btn:
    api_key = get_api_key()
    if not api_key:
        st.error("APIキーが見つかりません。`.streamlit/secrets.toml` に `GEMINI_API_KEY` を設定してください。")
        st.stop()

    rule_text = ""
    if uploaded_file:
        try:
            rule_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
            st.stop()
    elif rule_text_direct.strip():
        rule_text = rule_text_direct.strip()
    else:
        st.warning("就業規則ファイルをアップロードするか、テキストを入力してください。")
        st.stop()

    with st.spinner("Geminiが解析中です。しばらくお待ちください..."):
        try:
            client = genai.Client(api_key=api_key)
            prompt = build_prompt(rule_text, reality_memo)
            # Windows環境でのエンコードエラー対策: bytes経由でUTF-8を保証
            prompt = prompt.encode("utf-8").decode("utf-8")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            result = response.text
        except Exception as e:
            st.error(f"Gemini APIエラー: {e}")
            st.stop()

    st.divider()
    st.subheader("📋 解析結果")
    st.markdown(result)
    st.download_button(
        label="📥 結果をテキストでダウンロード",
        data=result,
        file_name="rule_adapt_result.txt",
        mime="text/plain",
    )
