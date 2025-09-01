import os
import json
import time
from datetime import datetime
import torch
import numpy as np
import pyaudio
import cv2
try:
    import serial  # optional
except Exception:  # serial is optional
    serial = None
from transformers import (
    AutoProcessor,
    SeamlessM4TModel,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# -------- Settings --------
model_id = "facebook/hf-seamless-m4t-medium"
sample_rate = 16000
channels = 1
record_seconds = 5  # change if you want longer/shorter capture
src_lang = "eng"    # input speech language code
tgt_lang = "eng"    # output text and TTS language code
ser_model_id = "superb/hubert-large-superb-er"  # speech emotion recognition model
face_model_id = "trpakov/vit-face-expression"   # facial emotion recognition model (HF)

# Face and robot settings
enable_face = True
webcam_index = 0

# Optional serial control (set to COM port string like "COM3" to enable)
serial_port = None
serial_baudrate = 115200

# Logging & privacy
log_dir = os.path.join(os.getcwd(), "logs")
log_file = os.path.join(log_dir, "interactions.jsonl")
os.makedirs(log_dir, exist_ok=True)
privacy_save_audio = False   # set True if you want to save raw audio
privacy_save_frames = True   # save face/frame images


def record_audio(duration_sec: int, rate: int, nchannels: int) -> np.ndarray:
    """Record mono audio from the default microphone and return float32 waveform in [-1,1]."""
    pa = pyaudio.PyAudio()
    frames = []
    stream = pa.open(format=pyaudio.paInt16, channels=nchannels, rate=rate, input=True, frames_per_buffer=1024)
    try:
        print(f"🎙 Speak now ({duration_sec}s)...")
        for _ in range(int(rate / 1024 * duration_sec)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # Convert to float32 mono waveform
    audio_bytes = b"".join(frames)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = (audio_int16.astype(np.float32) / 32768.0)  # normalize to [-1, 1]
    return audio_float


# -------- Load processor + model (single model handles S2TT and TTS) --------
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

if use_cuda:
    model = SeamlessM4TModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
else:
    model = SeamlessM4TModel.from_pretrained(model_id)
    model.to("cpu")


def transcribe(audio_np: np.ndarray) -> str:
    """Speech -> Text using SeamlessM4TModel (generate_speech=False)."""
    inputs = processor(audios=audio_np, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        out_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
    # decode first sample
    text_out = processor.decode(out_tokens[0].tolist()[0], skip_special_tokens=True)
    return text_out.strip()


def tts(text: str) -> np.ndarray:
    """Text -> Speech waveform using SeamlessM4TModel (returns (waveforms, lengths))."""
    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**inputs, tgt_lang=tgt_lang)
    waveforms = gen[0] if isinstance(gen, (list, tuple)) else gen
    audio = waveforms[0].cpu().numpy().squeeze().astype(np.float32)
    return audio


def play_audio(wave: np.ndarray, rate: int):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
    try:
        stream.write(wave.tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def capture_face_frame(timeout_sec: float = 4.0):
    """Capture a single frame with a detected face; returns (face_rgb, frame_rgb)."""
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        return None, None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start = time.time()
    face_rgb = None
    frame_rgb = None
    while time.time() - start < timeout_sec:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            break
    cap.release()
    return face_rgb, frame_rgb


def send_gesture_command(emotion: str):
    """Send a simple gesture/motion command over serial based on emotion (optional)."""
    if not serial_port or serial is None:
        return
    try:
        with serial.Serial(serial_port, serial_baudrate, timeout=1) as ser:
            # Example protocol: E:<EMOTION>\n  (adapt on microcontroller)
            cmd = f"E:{emotion}\n".encode()
            ser.write(cmd)
    except Exception:
        pass


def fuse_emotions(audio_emo: tuple[str, float] | None, face_emo: tuple[str, float] | None) -> tuple[str, float]:
    """Pick the most salient emotion between audio and face (by confidence, prefer non-neutral)."""
    candidates = []
    if audio_emo:
        candidates.append(("audio",) + audio_emo)
    if face_emo:
        candidates.append(("face",) + face_emo)
    if not candidates:
        return ("neutral", 0.0)
    # prefer non-neutral
    non_neutral = [c for c in candidates if c[1].lower() not in ("neutral", "neu")]  # label normalization
    pool = non_neutral if non_neutral else candidates
    src, label, conf = max(pool, key=lambda c: c[2])
    return (label, conf)


def log_interaction(payload: dict):
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    # Load SER model (prefer GPU if available)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ser_processor = AutoFeatureExtractor.from_pretrained(ser_model_id)
    ser_model = AutoModelForAudioClassification.from_pretrained(ser_model_id).to(device)
    # Load facial emotion model (optional)
    face_processor = None
    face_model = None
    if enable_face:
        try:
            face_processor = AutoImageProcessor.from_pretrained(face_model_id)
            face_model = AutoModelForImageClassification.from_pretrained(face_model_id).to(device)
        except Exception:
            enable_face = False  # disable gracefully

    def detect_emotion(audio_np: np.ndarray):
        inputs = ser_processor(audio_np, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            label = ser_model.config.id2label.get(pred_id, str(pred_id))
            conf = float(probs[pred_id])
        return label, conf

    def detect_face_emotion(face_rgb: np.ndarray):
        if face_processor is None or face_model is None or face_rgb is None:
            return None
        inputs = face_processor(images=face_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = face_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            label = face_model.config.id2label.get(pred_id, str(pred_id))
            conf = float(probs[pred_id])
        return label, conf

    def compose_consoling_message(emotion_label: str, transcript: str | None) -> str:
        e = emotion_label.lower()
        base = ""
        if "sad" in e:
            base = (
                "I'm sorry you're feeling down. You're not alone, and it's okay to take things one step at a time. "
                "Try a deep breath with me: in for four, hold for four, and out for six."
            )
        elif "angry" in e or "frustrat" in e:
            base = (
                "I can hear the frustration. It's valid to feel this way. Let's pause for a moment and release some tension with a slow breath."
            )
        elif "fear" in e or "anx" in e:
            base = (
                "It sounds like you're feeling anxious. You're safe right now. Let's ground together—notice three things you can see, two you can touch, and one you can hear."
            )
        elif "happy" in e or "joy" in e:
            base = "You sound happy—that's wonderful. I’m glad to share this moment with you."
        elif "neutral" in e or "calm" in e:
            base = "Thanks for sharing. I'm here with you if you'd like to talk more."
        elif "surprise" in e:
            base = "That sounded surprising. Take a moment to notice how your body feels and let it settle."
        else:
            base = "I'm listening. Whatever you're feeling is valid, and I'm here for you."

        if transcript and len(transcript) > 0:
            return f"You said: {transcript}. {base}"
        return base

    # 1) Record
    audio_in = record_audio(record_seconds, sample_rate, channels)
    raw_face, full_frame = capture_face_frame() if enable_face else (None, None)
    if privacy_save_frames:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if raw_face is not None:
            cv2.imwrite(os.path.join(log_dir, f"face_{ts}.jpg"), cv2.cvtColor(raw_face, cv2.COLOR_RGB2BGR))
        elif full_frame is not None:
            cv2.imwrite(os.path.join(log_dir, f"frame_{ts}.jpg"), cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

    # 2) Speech -> Text
    print("📝 Transcribing...")
    text = transcribe(audio_in)
    print(f"You said: {text}")

    # 3) Emotion detection
    print("💗 Detecting emotion...")
    audio_emotion = detect_emotion(audio_in)
    face_emotion = detect_face_emotion(raw_face) if raw_face is not None else None
    fused_label, fused_conf = fuse_emotions(audio_emotion, face_emotion)
    print(f"Audio emotion: {audio_emotion}, Face emotion: {face_emotion}, Fused: {fused_label} ({fused_conf:.2f})")

    # 4) Compose supportive reply and speak
    reply = compose_consoling_message(fused_label, text)
    send_gesture_command(fused_label)
    print("🔊 Speaking a supportive message...")
    audio_out = tts(reply)
    play_audio(audio_out, sample_rate)
    # Logging
    ts = datetime.utcnow().isoformat() + "Z"
    log_interaction({
        "timestamp": ts,
        "transcript": text,
        "audio_emotion": audio_emotion,
        "face_emotion": face_emotion,
        "fused_emotion": [fused_label, fused_conf],
        "reply": reply,
    })
    print("Done.")