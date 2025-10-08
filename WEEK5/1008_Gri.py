import torch
import torchaudio
from pathlib import Path

# 실행마다 결과가 달라지지 않도록 시드를 고정합니다.
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tacotron2 + Griffin-Lim 문자 기반 번들을 불러옵니다.
bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

# 합성할 영어 문장(필요하면 수정 가능)
text = (
    "Hello, I'm Youngjin Lee. I'm currently studying Computer Science at Chungbuk National University."
    "I've a focus on artificial intelligence, machine learning, and backend development."
    "Recently, I've become interested in Nietzsche's philosophy and am reading Beyond Good and Evil."
)

# Tacotron2로 멜 스펙트로그램을 만들고, Griffin-Lim 보코더로 파형을 복원합니다.
with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)

# WAV로 저장할 폴더를 만들고 결과 음성을 기록합니다.
output_dir = Path("outputs/tacotron2_self_intro")
output_dir.mkdir(parents=True, exist_ok=True)
waveform = waveforms.squeeze(0).cpu()
sample_rate = getattr(vocoder, "sample_rate", 22050)
torchaudio.save(
    str(output_dir / "1008_gri.wav"),
    waveform.unsqueeze(0),
    sample_rate=sample_rate,
    format="wav",
)
print(f"✓ Saved Griffin-Lim waveform file: {output_dir / '1008_gri.wav'}")
