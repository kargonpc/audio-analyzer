import pytest
from pathlib import Path
from organizer import AudioAnalyzer 

TEST_AUDIO_PATH = Path("tests/wa.wav")

def test_bpm_detection():
    assert TEST_AUDIO_PATH.exists(), "Тестовый аудиофайл не найден!"
    
    result = AudioAnalyzer.analyze(TEST_AUDIO_PATH)
    
    assert result.error is None
    
    assert result.bpm_main == 140 or result.bpm_alt == 140

def test_key_detection():
    result = AudioAnalyzer.analyze(TEST_AUDIO_PATH)
    
    # Проверяем, что скрипт нашел либо CMin, либо параллельный мажор (EbMaj)
    assert result.key_main in ["CMin", "D#Maj", "EbMaj"]

def test_bandpass_filter_shape():
    import numpy as np
    # Генерируем фейковый белый шум (1 секунда)
    noise = np.random.normal(0, 1, 44100)
    
    # Пропускаем через твой фильтр
    filtered = AudioAnalyzer._apply_bandpass_filter(noise, 44100)
    
    # Длина массива должна остаться прежней
    assert len(noise) == len(filtered)