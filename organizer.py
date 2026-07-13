import click
import librosa
import numpy as np
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Optional
import warnings
import json
import concurrent.futures

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from scipy.signal import butter, filtfilt

# Подавляем системные предупреждения при чтении аудио
warnings.filterwarnings("ignore")
console = Console()

# --- 1. СТРОГАЯ ТИПИЗАЦИЯ (Dataclasses) ---
@dataclass
class AudioAnalysisResult:
    original_path: str
    filename: str
    bpm_main: int
    bpm_alt: int
    key_main: str
    key_alt: str
    new_name: str
    error: Optional[str] = None

# --- 2. ООП: МАТЕМАТИКА И DSP ---
class AudioAnalyzer:
    """Класс, отвечающий исключительно за математический анализ звука (DSP)."""
    
    PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MAJ_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MIN_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    @staticmethod
    def _apply_bandpass_filter(y: np.ndarray, sr: int, lowcut=60.0, highcut=2000.0) -> np.ndarray:
        """Срезает саб-бас (грязь от 808-х) и высокие частоты (тарелки) для чистого анализа."""
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, y)

    @classmethod
    def analyze(cls, file_path: Path) -> AudioAnalysisResult:
        try:
            # ПРОИЗВОДИТЕЛЬНОСТЬ: Грузим только 30 секунд из середины трека!
            # Это ускоряет работу скрипта в 5-10 раз.
            y, sr = librosa.load(str(file_path), sr=44100, mono=True, offset=15.0, duration=30.0)
            
            # 1. Анализ BPM
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None, start_bpm=130.0)
            
            bpm_counts = Counter([int(round(t)) for t in tempos])
            top_bpms = bpm_counts.most_common(2)
            bpm_main = top_bpms[0][0]
            bpm_alt = top_bpms[1][0] if len(top_bpms) > 1 else bpm_main

            # 2. Анализ тональности
            # Применяем фильтр и отделяем гармонию
            y_filtered = cls._apply_bandpass_filter(y, sr)
            y_harmonic, _ = librosa.effects.hpss(y_filtered)
            
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            chroma_sum = np.sum(chroma, axis=1)
            
            correlations = []
            for i in range(12):
                maj_rot = np.roll(cls.MAJ_PROFILE, i)
                min_rot = np.roll(cls.MIN_PROFILE, i)
                corr_maj = np.corrcoef(chroma_sum, maj_rot)[0, 1]
                corr_min = np.corrcoef(chroma_sum, min_rot)[0, 1]
                
                correlations.append((corr_maj, f"{cls.PITCHES[i]}Maj", "Maj", i))
                correlations.append((corr_min, f"{cls.PITCHES[i]}Min", "Min", i))
                
            correlations.sort(key=lambda x: x[0], reverse=True)
            best_mode = correlations[0][2]
            best_pitch_idx = correlations[0][3]
            key_main = correlations[0][1]
            
            # Параллельная тональность
            if best_mode == "Maj":
                rel_idx = (best_pitch_idx - 3) % 12
                key_alt = f"{cls.PITCHES[rel_idx]}Min"
            else:
                rel_idx = (best_pitch_idx + 3) % 12
                key_alt = f"{cls.PITCHES[rel_idx]}Maj"

            # Формируем новое имя
            name, ext = file_path.stem, file_path.suffix
            new_name = f"{name}_{bpm_main}bpm_{key_main}{ext}" if "bpm" not in name.lower() else file_path.name

            return AudioAnalysisResult(
                original_path=str(file_path), filename=file_path.name,
                bpm_main=bpm_main, bpm_alt=bpm_alt, key_main=key_main, key_alt=key_alt,
                new_name=new_name
            )
            
        except Exception as e:
            return AudioAnalysisResult(
                original_path=str(file_path), filename=file_path.name,
                bpm_main=0, bpm_alt=0, key_main="", key_alt="",
                new_name=file_path.name, error=str(e)
            )

# Топ-уровневая функция для обхода проблем с сериализацией (pickle) при мультипроцессинге
def analyze_worker(file_path: Path) -> AudioAnalysisResult:
    return AudioAnalyzer.analyze(file_path)

# --- 3. ООП: МЕНЕДЖЕР ФАЙЛОВ И ДАННЫХ ---
class LibraryManager:
    @staticmethod
    def rename_file(result: AudioAnalysisResult) -> None:
        if result.error or result.filename == result.new_name:
            return
        
        old_path = Path(result.original_path)
        new_path = old_path.parent / result.new_name
        old_path.rename(new_path)
        result.original_path = str(new_path) # Обновляем путь после переименования

    @staticmethod
    def export_to_json(results: List[AudioAnalysisResult], output_path: str = "library_index.json"):
        data = [asdict(r) for r in results if not r.error]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        console.print(f"\n[bold magenta]💾 База данных сэмплов сохранена в: {output_path}[/bold magenta]")

# --- 4. ИНТЕРФЕЙС И ОРКЕСТРАЦИЯ ---
@click.command()
@click.argument('target_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option('--rename', is_flag=True, help='Физически переименовать файлы')
@click.option('--export-json', is_flag=True, help='Сохранить результаты в JSON базу данных')
def main(target_path, rename, export_json):
    """Pro-утилита для анализа сэмплов: BPM, Тональность, Мультипроцессинг, DSP."""
    valid_extensions = {'.wav', '.mp3', '.aiff', '.flac', '.ogg', '.m4a'}
    target = Path(target_path)
    
    files_to_process = [target] if target.is_file() and target.suffix.lower() in valid_extensions else \
                       [p for p in target.rglob('*') if p.is_file() and p.suffix.lower() in valid_extensions]
    
    if not files_to_process:
        console.print("[bold red]Подходящие аудиофайлы не найдены![/bold red]")
        return
        
    console.print(f"[bold blue]🚀 Найдено файлов: {len(files_to_process)}. Запуск мультипроцессорного анализа...[/bold blue]\n")
    
    results: List[AudioAnalysisResult] = []
    
    # МУЛЬТИПРОЦЕССИНГ: Используем все ядра процессора!
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Анализ аудио (Multicore)...", total=len(files_to_process))
        
        # Запускаем пул процессов (по дефолту = количеству ядер CPU)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(analyze_worker, f): f for f in files_to_process}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Логика UI
                if result.error:
                    progress.console.print(f"[red]❌ Ошибка ({result.filename}): {result.error}[/red]")
                else:
                    if rename:
                        LibraryManager.rename_file(result)
                        progress.console.print(f"✅ [green]{result.new_name}[/green]")
                    else:
                        bpm_display = f"{result.bpm_main}" if result.bpm_main == result.bpm_alt else f"{result.bpm_main} ({result.bpm_alt})"
                        progress.console.print(f"🎵 [cyan]{result.filename}[/cyan] --> [bold green]{bpm_display} BPM | {result.key_main}[/bold green] (или {result.key_alt})")
                
                progress.advance(task)

    # Экспорт базы данных
    if export_json:
        LibraryManager.export_to_json(results)

if __name__ == '__main__':
    main()