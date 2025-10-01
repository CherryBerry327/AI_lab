# 참고 링크(튜토리얼):
#   - https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
#   - https://docs.pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html

import glob
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

# 기본 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "WAV2VEC2_ASR_BASE_960H"  # torchaudio 사전학습 번들
USE_VAD = True          # 앞/뒤 무음 컷 할지
SAVE_TXT = True         # 인식 결과를 txt로 저장할지

# /content 밑에 있는 오디오 파일들 전부 찾기
FILES = glob.glob("/content/*.wav")



# 유틸 함수
def load_audio_to_model_sr(path: str, target_sr: int):
    """파일 로드 → 모노 → 모델 SR로 리샘플 → 간단 레벨 정규화"""
    wav, sr = torchaudio.load(path)         # (채널, 샘플)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True) # 스테레오 → 모노
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav) # 모델 샘플레이트로 변환
        sr = target_sr
    # 너무 작거나 큰 레벨 방지(간단 RMS 기반)
    rms = wav.pow(2).mean().sqrt().clamp(min=1e-6)
    wav = (wav / rms) * 0.1
    return wav, sr

def trim_silence_vad(wav: torch.Tensor, sr: int):
    """sox 'silenceremove'로 앞/뒤 무음 잘라내기(가능한 경우만)"""
    effects = [
        ["gain", "-n"],                          # 클리핑 방지용
        ["silenceremove", "1", "0.1", "1%",     # 앞: 0.1초 이상 1% 이하 레벨이면 컷
         "-1", "0.1", "1%"],                    # 뒤: 동일 기준
    ]
    try:
        out, out_sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        if out.numel() == 0:
            return wav, sr
        return out, out_sr
    except Exception:
        return wav, sr

def ctc_greedy_decode(emission: torch.Tensor, labels):
    """프레임별 argmax → blank/중복 제거 → 문자열"""
    with torch.no_grad():
        probs = emission.softmax(dim=-1)        # (T, vocab)
        idx = probs.argmax(dim=-1).tolist()     # 가장 높은 확률의 토큰 인덱스
    blank = 0
    last = None
    tokens = []
    for i in idx:
        if i == blank:
            last = None
            continue
        if i != last:
            tokens.append(i)
            last = i
    text = "".join(labels[i] for i in tokens)
    return text.replace("|", " ").strip()


# =========================
# 모델 로드
# =========================
# torchaudio 사전학습 번들 불러오기
BUNDLE = getattr(torchaudio.pipelines, MODEL_NAME)
MODEL = BUNDLE.get_model().to(DEVICE).eval()
LABELS = BUNDLE.get_labels()
MODEL_SR = BUNDLE.sample_rate  # 보통 16000Hz


# =========================
# 메인 처리
# =========================
if not FILES:
    print(">>> /content/*.wav 에 오디오 파일을 넣어주세요.")
else:
    for path in FILES:
        base = Path(path).stem
        print(f"\n -> {base}")

        # 1) 로드 + 모델 SR로 리샘플
        wav, sr = load_audio_to_model_sr(path, MODEL_SR)

        # 2) (옵션) 앞/뒤 무음 컷
        if USE_VAD:
            wav, sr = trim_silence_vad(wav, sr)

        # 3) ASR 추론
        with torch.inference_mode():
            emissions, _ = MODEL(wav.to(DEVICE))   # 입력 형태: (1, time)
        emission = emissions[0].cpu()              # (time, vocab)
        text = ctc_greedy_decode(emission, LABELS)

        # 4) 출력 + (옵션) 파일 저장
        print("인식 결과:")
        print(text)
        if SAVE_TXT:
            out_txt = f"/content/{base}.txt"
            Path(out_txt).write_text(text, encoding="utf-8")
            print(f"-> 저장됨: {out_txt}")

# 결과는 /content/{파일명}.txt 로 저장되어 확인할 수 있다.
