import whisper

# 初始化 Whisper
synthesizer = whisper.WhisperTTS()

# 合成语音
text = "你好，欢迎使用 Whisper 进行语音合成。"
synthesizer.synthesize(text)
