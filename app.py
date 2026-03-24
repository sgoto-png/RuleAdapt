import os
import re
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import fitz  # PyMuPDF
import docx

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 知識ベースを起動時に読み込む
def load_knowledge_base():
    kb_file = KNOWLEDGE_DIR / "career_up_qa.txt"
    if kb_file.exists():
        with open(kb_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

KNOWLEDGE_BASE = load_knowledge_base()
# Geminiに送るには長すぎる場合があるため、最初の30000文字に絞る
KNOWLEDGE_BASE_TRIMMED = KNOWLEDGE_BASE[:30000]


def get_gemini_client(api_key: str):
    if not api_key:
        raise ValueError("GEMINI_API_KEY が設定されていません")
    return genai.Client(api_key=api_key)


def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    texts = [page.get_text() for page in doc]
    return "\n".join(texts)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_rule_text(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".text"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"対応していないファイル形式です: {ext}")


def build_analysis_prompt(rule_text: str, reality_memo: str) -> str:
    prompt = f"""あなたは社会保険労務士の専門アシスタントです。
キャリアアップ助成金（正社員化コース）の支給要領に精通しており、就業規則の問題点を発見・修正提案する専門家です。

## 参考知識：キャリアアップ助成金Q&A（抜粋）
{KNOWLEDGE_BASE_TRIMMED}

---

## 分析対象：就業規則テキスト
{rule_text[:8000]}

---

## 実態メモ（新人スタッフからの現場情報）
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
    return prompt


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # APIキーチェック
    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "Gemini APIキーを入力してください。"}), 400

    # 実態メモ
    reality_memo = request.form.get("reality_memo", "").strip()

    # ファイルアップロード
    rule_file = request.files.get("rule_file")
    rule_text_direct = request.form.get("rule_text", "").strip()

    rule_text = ""
    if rule_file and rule_file.filename:
        suffix = Path(rule_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            rule_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            rule_text = extract_rule_text(tmp_path, rule_file.filename)
        except Exception as e:
            return jsonify({"error": f"ファイル読み込みエラー: {str(e)}"}), 400
        finally:
            os.unlink(tmp_path)
    elif rule_text_direct:
        rule_text = rule_text_direct

    if not rule_text:
        return jsonify({"error": "就業規則ファイルをアップロードするか、テキストを入力してください。"}), 400

    # Gemini呼び出し
    try:
        client = get_gemini_client(api_key)
        prompt = build_analysis_prompt(rule_text, reality_memo)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        result_text = response.text
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Gemini APIエラー: {str(e)}"}), 500

    return jsonify({"result": result_text})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
