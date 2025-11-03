from datasets import load_dataset, Audio
import os

print("🔍 Loading dataset...")
ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")

print("✅ Dataset loaded.")
print(ds)
print("\n[Dataset features]")
print(ds.features)

# 오디오 경로 확인
missing = 0
exists = 0

print("\n🔎 Checking audio file paths...")
for i in range(10):
    print(ds[i]["audio"])  # 10개 샘플 미리보기

for example in ds:
    path = example["audio"]["path"]
    if path and os.path.exists(path):
        exists += 1
    else:
        missing += 1

print(f"\n✅ Total samples: {len(ds)}")
print(f"📁 Existing .wav files: {exists}")
print(f"❌ Missing .wav files: {missing}")

if missing > 0:
    print("\n⚠️ Many missing audio files detected. The dataset is likely metadata-only.")
else:
    print("\n✅ All audio files are present. Dataset is ready for training.")
