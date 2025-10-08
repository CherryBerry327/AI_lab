import torch
import torchaudio
from pathlib import Path

# 실행마다 결과가 다르지 않도록 시드 고정
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 문자 기반 Tacotron2 (WaveRNN 번들에서 Tacotron2 모델만 사용)
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device).eval()

# WaveGlow 보코더는 TorchHub에서 불러옵니다.
waveglow = torch.hub.load(
    "nvidia/DeepLearningExamples:torchhub",
    "nvidia_waveglow",
    trust_repo=True,
)
waveglow = waveglow.remove_weightnorm(waveglow).to(device).eval()

# 합성할 문장
text = (
    "Hello, I'm Youngjin Lee. I'm currently studying Computer Science at Chungbuk National University."
    "I've a focus on artificial intelligence, machine learning, and backend development."
    "Recently, I've become interested in Nietzsche's philosophy and am reading Beyond Good and Evil."
)

# Tacotron2로 멜 스펙트로그램을 생성하고 WaveGlow로 파형을 복원
with torch.inference_mode():
    tokens, lengths = processor(text)
    tokens = tokens.to(device)
    lengths = lengths.to(device)
    mel, mel_lengths, _ = tacotron2.infer(tokens, lengths)
    audio = waveglow.infer(mel, sigma=0.7)

# WaveGlow 출력 정규화 및 CPU 텐서 변환
audio = audio.squeeze(0).cpu()
audio = audio / audio.abs().max()

# 결과 저장
output_dir = Path("outputs/tacotron2_self_intro")
output_dir.mkdir(parents=True, exist_ok=True)
torchaudio.save(
    str(output_dir / "1008_waveglow.wav"),
    audio.unsqueeze(0),
    sample_rate=22050,
    format="wav",
)

print(f"✓ Saved WaveGlow waveform file: {output_dir / '1008_waveglow.wav'}")
