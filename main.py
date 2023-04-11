import time
import torch
import wave
import pyaudio
from faster_whisper import WhisperModel
import requests
import json

def record_audio_from_microphone(filename, duration):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

    frames = []
    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

def transcribe_audio(filename, model):
    start_time = time.time()
    segments, _ = model.transcribe(filename, beam_size=5, language="zh")
    transcription = " ".join([segment.text for segment in segments])
    elapsed_time = time.time() - start_time
    return transcription, elapsed_time

def translate_text(text, target_lang):
    start_time = time.time()
    api_key = 'fbfd0fc0-627a-9f90-c975-10571690a3a2:fx'
    url = 'https://api-free.deepl.com/v2/translate'
    headers = {'Authorization': f'DeepL-Auth-Key {api_key}'}
    data = {'text': text, 'target_lang': target_lang}

    response = requests.post(url, headers=headers, data=data)
    translated_response = json.loads(response.text)
    translation = translated_response['translations'][0]['text']
    elapsed_time = time.time() - start_time
    return translation, elapsed_time

def main():
    model_size = "large-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_size, device=device, compute_type="float16")

    target_lang = "zh"  # 输出目标翻译语言，例如英语（en）

    while True:
        audio_filename = "temp_audio.wav"
        record_audio_from_microphone(audio_filename, 10)

        transcription, transcription_time = transcribe_audio(audio_filename, model)
        translation, translation_time = translate_text(transcription, target_lang)

        print(f"[{transcription_time:.2f}s] 原文：{transcription}")
        print(f"[{translation_time:.2f}s] 翻译：{translation}")

if __name__ == "__main__":
    main()
