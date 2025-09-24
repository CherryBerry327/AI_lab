import glob
import torchaudio, torch
import matplotlib.pyplot as plt
from pathlib import Path

# /content 밑에 있는 오디오 파일들 전부 찾기
files = glob.glob("/content/*.wav")

# 파일마다 파형과 스펙트로그램 출력
for path in files:
    waveform, sr = torchaudio.load(path)
    base = Path(path).stem
    print(f"\n -> {base} (sr={sr})")

    # 파형 그리기
    plt.figure(figsize=(10,3))
    plt.plot(waveform.t().numpy())
    plt.title(f"Waveform - {base}")
    plt.show()

    # 스펙트로그램 그리기
    spec = torchaudio.transforms.Spectrogram()(waveform)
    plt.figure(figsize=(10,3))
    plt.imshow(spec.log2()[0,:,:].numpy(), origin="lower", aspect="auto")
    plt.title(f"Spectrogram - {base}")
    plt.colorbar()
    plt.show()

# 리샘플링 함수 정의
def resample(wav, orig_sr, new_sr):
    rs = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    return rs(wav)

# 각 파일에 대해 다운샘플링(8kHz)과 업샘플링(32kHz) 수행
for path in files:
    waveform, sr = torchaudio.load(path)
    base = Path(path).stem

    # 다운샘플링 → 8kHz
    down = resample(waveform, sr, 8000)
    torchaudio.save(f"/content/{base}_down8k.wav", down, 8000)
    print(f"{base} saved (down 8kHz)")

    # 업샘플링 → 32kHz
    up = resample(waveform, sr, 32000)
    torchaudio.save(f"/content/{base}_up32k.wav", up, 32000)
    print(f"{base} saved (up 32kHz)")


# 결과는 파일 내 자동으로 생성되어 들어볼 수 있다.
