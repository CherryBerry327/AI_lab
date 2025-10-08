import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt
from pathlib import Path

# 실행할 때마다 결과가 크게 달라지지 않도록 시드를 고정합니다.
torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 문자 기반 Tacotron2 + WaveRNN 번들을 사용합니다. (폰 기반 대신 CHAR 모델)
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

# 음성으로 만들 자기소개 문장입니다.
text = (
    "Hello, I'm Youngjin Lee. I'm currently studying Computer Science at Chungbuk National University."
    "I've a focus on artificial intelligence, machine learning, and backend development."
    "Recently, I've become interested in Nietzsche's philosophy and am reading Beyond Good and Evil."
)

# Tacotron2가 멜 스펙트로그램을 생성하고 WaveRNN이 파형을 복원합니다.
with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)


def plot(waveforms, spec, sample_rate):
    """멜 스펙트로그램과 파형을 간단히 시각화하고 오디오 위젯을 반환합니다."""
    waveforms = waveforms.cpu().detach()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(waveforms[0])
    ax1.set_xlim(0, waveforms.size(-1))
    ax1.grid(True)
    ax2.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    plt.tight_layout()
    return IPython.display.Audio(waveforms[0:1], rate=sample_rate)


# 노트북 환경에서 실행하면 파형과 스펙트로그램을 바로 확인할 수 있습니다.
plot(waveforms, spec, vocoder.sample_rate)

# 합성된 결과를 프로젝트 폴더에 WAV 파일로 저장합니다.
output_dir = Path("outputs/tacotron2_self_intro")
output_dir.mkdir(parents=True, exist_ok=True)
waveform = waveforms.squeeze(0).cpu()
sample_rate = getattr(vocoder, "sample_rate", 22050)
torchaudio.save(
    str(output_dir / "1008_wavernn.wav"),
    waveform.unsqueeze(0),
    sample_rate=sample_rate,
    format="wav",
)
print(f"✓ Saved waveform file: {output_dir / '1008_wavernn.wav'}")
