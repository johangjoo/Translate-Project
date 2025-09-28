# transcribe_diarize.py
import os, argparse, pathlib
from pathlib import Path
import torch, whisperx
from whisperx.diarize import DiarizationPipeline

# Windows 전용: PyTorch DLL 경로 우선
try:
    os.add_dll_directory(str(pathlib.Path(torch.__file__).parents[1] / "lib"))
except Exception:
    pass

# 옵션: 속도 향상(TF32). 필요 없으면 주석 처리.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def mmss(t):
    t = 0.0 if not isinstance(t, (int, float)) or t != t or t < 0 else float(t)
    m = int(t // 60)
    s = int(round(t - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"

def write_lines(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def build_diar_pipeline(token: str, prefer_device: str):
    try:
        return DiarizationPipeline(use_auth_token=token, device=prefer_device)
    except OSError as e:
        print(f"[WARN] GPU diarization 실패: {e}\n[INFO] CPU로 폴백.")
        return DiarizationPipeline(use_auth_token=token, device="cpu")

def looks_blackwell(gpu_name: str):
    n = gpu_name.lower()
    keys = ["rtx 50", "rtx5", "5070", "5080", "5090", "blackwell", "gb2", "gb20"]
    return any(k in n for k in keys)

def process_one(fp: Path, out_dir: Path, model, diar_pipe, device: str, batch: int,
                min_spk, max_spk, load_model_fn, compute_type: str):
    audio = whisperx.load_audio(str(fp))

    # 1) ASR(+언어감지). INT8 경로(cuBLAS) 실패 시 FP16으로 1회 자동 재시도.
    try:
        asr = model.transcribe(audio, batch_size=batch)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("cublas_status_not_supported" in msg or "cublas" in msg) and "int8" in (compute_type or ""):
            print("[INFO] cuBLAS INT8 문제 감지. FP16으로 재시도.")
            model = load_model_fn("float16")
            asr = model.transcribe(audio, batch_size=batch)
        else:
            raise

    lang = asr.get("language", "ko")

    # 2) 정렬
    align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
    asr_aligned = whisperx.align(
        asr["segments"], align_model, meta, audio, device, return_char_alignments=False
    )

    # 3) 화자분리
    diar_kwargs = {}
    if min_spk is not None:
        diar_kwargs["min_speakers"] = min_spk
    if max_spk is not None:
        diar_kwargs["max_speakers"] = max_spk
    diar_segments = diar_pipe(audio, **diar_kwargs)
    asr_spk = whisperx.assign_word_speakers(diar_segments, asr_aligned)

    # 4) 저장: [mm:ss] 화자N : 텍스트
    spk_map, lines = {}, []
    for seg in sorted(asr_spk["segments"], key=lambda x: x.get("start", 0.0)):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        spk = seg.get("speaker", "UNK")
        if spk not in spk_map:
            spk_map[spk] = f"화자{len(spk_map) + 1}"
        lines.append(f"[{mmss(seg.get('start', 0.0))}] {spk_map[spk]} : {txt}")

    out_path = out_dir / f"{fp.stem}_transcript_{lang}.txt"
    write_lines(out_path, lines)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="WhisperX + diarization 배치")
    ap.add_argument("--src", default="video_input", help="mp4 파일 또는 폴더")
    ap.add_argument("--out_dir", default="input", help="출력 폴더")
    ap.add_argument("--whisper_model", default="large-v3", help="Whisper 모델명")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)
    ap.add_argument("--hf_token", default=None, help="Hugging Face read 토큰")
    ap.add_argument("--force_cpu_diar", action="store_true", help="화자분리만 CPU 강제")
    ap.add_argument("--compute_type", default=None, help='CUDA: float16|bfloat16|float32|int8_float16 등')
    ap.add_argument("--ext", default="*.mp4", help="검색 확장자 패턴 (예: *.mp4)")
    args = ap.parse_args()

    token = (
        args.hf_token
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    assert token and token.startswith("hf_"), "HF 토큰 필요. 게이트드 레포 동의 포함."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else ""
    is_blackwell = device == "cuda" and looks_blackwell(gpu_name)

    # 기본 정밀도 선택: Blackwell(RTX 50xx)에서는 INT8 계열 회피
    if args.compute_type:
        compute_type = args.compute_type
    else:
        if device == "cuda":
            compute_type = "float16" if is_blackwell else "int8_float16"
        else:
            compute_type = "int8"

    def load_model_with(ct):
        return whisperx.load_model(args.whisper_model, device=device, compute_type=ct)

    # 모델 로드
    model = load_model_with(compute_type)

    diar_device = "cpu" if args.force_cpu_diar else device
    diar_pipe = build_diar_pipeline(token, diar_device)

    # 입력 수집
    src_path = Path(args.src)
    files = [src_path] if src_path.is_file() else sorted(src_path.glob(args.ext))
    assert files, f"{src_path}에서 {args.ext}를 찾지 못함."

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        try:
            outp = process_one(
                fp, out_dir, model, diar_pipe, device, args.batch_size,
                args.min_speakers, args.max_speakers, load_model_with, compute_type
            )
            print(f"OK: {fp.name} -> {outp}")
        except Exception as e:
            print(f"ERR: {fp.name}: {e}")

if __name__ == "__main__":
    main()
