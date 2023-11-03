import os
import time
import concurrent.futures
from faster_whisper import WhisperModel
from Constants import FileConstants as fileConst

def transcribe_audio(file_path, model):
    segments, info = model.transcribe(file_path)

    transcription_text = [segment.text for segment in segments]
    full_transcription = ''.join(transcription_text)

    output_file = os.path.splitext(file_path)[0] + "_transcription.txt"

    with open(output_file, 'w') as file:
        file.write(full_transcription)

    print(f"Transcription for {file_path} saved to '{output_file}'")

if __name__ == "__main__":
    start = time.time()
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="int8", device_index=[0, 1, 2, 3])

    audio_folder = fileConst.AUDIO_FOLDER_PATH
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".mp3")]

    num_workers = 4

    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        futures = {executor.submit(transcribe_audio, file_path, model): file_path for file_path in audio_files}

        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Transcription for {file_path} generated an error: {e}")

    end = time.time()
    print(f'Total time: {end - start}')
