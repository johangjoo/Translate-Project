# gpt_qwen.py
import re, json, requests
from pathlib import Path

INPUT_DIR  = Path("./input")
OUTPUT_DIR = Path("./output_qwen3")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen3:8b-q8_0"
TEMPERATURE = 0.0
TIMEOUT     = 180
MAX_CHARS   = 2000   # 더 작은 청크로 지연 감소

PROMPT_TPL = """You are a bilingual Korean↔Japanese translator for SONG LYRICS.
Rules:
- Process EACH LINE independently.
- Detect per-line language (Korean or Japanese) and translate into the OPPOSITE language.
- Ignore timestamps and speaker labels entirely. Translate ONLY lyrical content.
- Preserve quotes, punctuation, numbers, and LINE COUNT. Output must have EXACTLY the same number of lines as input.
- No explanations, no notes, no romanization. Plain text only.

[INPUT]
{chunk}
"""

# ----- IO -----
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

# ----- Pre-clean: timestamps / speaker labels 제거 -----
ts_pat = re.compile(r'^\s*(?:\[|\()?<?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?>?(?:\]|\))?\s*')
spk_pat = re.compile(r'^\s*[A-Za-z0-9가-힣一-龥ぁ-んァ-ヶー（）\[\]<>]{1,12}\s*:\s*')

def clean_line(line: str) -> str:
    s = ts_pat.sub('', line)         # 선행 타임스탬프 제거
    s = spk_pat.sub('', s)           # 선행 화자 라벨 제거
    return s

def clean_text(text: str) -> str:
    return "\n".join(clean_line(ln) for ln in text.splitlines())

# ----- Chunking -----
def split_chunks_by_chars(text: str, max_chars: int = MAX_CHARS):
    buf, size = [], 0
    for line in text.splitlines():
        ln = line + "\n"
        if size + len(ln) > max_chars and buf:
            yield "".join(buf)
            buf, size = [ln], len(ln)
        else:
            buf.append(ln); size += len(ln)
    if buf:
        yield "".join(buf)

# ----- Ollama -----
def warmup(model: str):
    payload = {"model": model, "prompt": "ok", "stream": False, "keep_alive": "10m",
               "options": {"temperature": 0.0}}
    try: requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT).raise_for_status()
    except: pass

def gen_stream_to_str(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": True, "keep_alive": "10m",
               "options": {"temperature": TEMPERATURE, "repeat_penalty": 1.05}}
    out = []
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line: 
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj:
                out.append(obj["response"])
            if obj.get("done"):
                break
    return "".join(out)

def translate_file(in_path: Path) -> str:
    raw = read_text(in_path).rstrip("\n")
    text = clean_text(raw)                   # 사전 정리
    parts = []
    for chunk in split_chunks_by_chars(text):
        prompt = PROMPT_TPL.format(chunk=chunk)
        parts.append(gen_stream_to_str(MODEL, prompt))
    return "".join(parts)

def main():
    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        print("input 폴더에 .txt 없음"); return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup(MODEL)
    for f in files:
        out_text = translate_file(f)
        out_path = OUTPUT_DIR / f"{f.stem}.koja.txt"
        write_text(out_path, out_text)
        print(f"OK  {f.name} -> {out_path}")

if __name__ == "__main__":
    main()
