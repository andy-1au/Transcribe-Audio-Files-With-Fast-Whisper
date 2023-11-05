import os
import time
import concurrent.futures
import math
from faster_whisper import WhisperModel
from Constants import FileConstants as fileConst

def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600) # Gets the total hours with the remainder being seconds 
    minutes, seconds = divmod(remainder, 60) # Gets the total minutes from the remainder of seconds, with the remainder being seconds
    milliseconds = math.floor((seconds % 1) * 1000) # Get the miliseconds from remaining fractional seconds
    output = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}' # Formatting string 
    return output

def transcribe_audio(file_path, model):
    segments, info = model.transcribe(file_path)
    output_file = os.path.splitext(file_path)[0] + "_transcription.vtt"
    count = 0
    with open(output_file, 'w') as file:
        try:
            file.write(f'{fileConst.VTT_HEADER}\n\n') # Add the WEBVTT header
            for segment in segments:
                count += 1
                duration = f'{convert_seconds_to_hms(segment.start)} --> {convert_seconds_to_hms(segment.end)}\n' # Both segment.start and segment.end are returned in seconds
                text = f'{segment.text.lstrip()}\n\n' # Removing any leading whitespaces, tabs, etc and double line break for formatting
                try:
                    file.write(f'{count}\n{duration}{text}') # Write formatted string to the file
                    print(f'{duration}{text}', end='') # Keeps the duration and text together without separating them on an empty line
                except Exception as e: 
                    print(f'Error occurred: {e}')
                    break
        except Exception as e:
            print(f'Error occurred: {e}')

    print(f"Transcription for {file_path} saved to '{output_file}'")

if __name__ == "__main__":
    start = time.time()

    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="int8", device_index=[0, 1, 2, 3])

    audio_folder = fileConst.AUDIO_FOLDER_PATH
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".mp3")]

    num_workers = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(transcribe_audio, file_path, model): file_path for file_path in audio_files}

        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Transcription for {file_path} generated an error: {e}")

    end = time.time()
    print(f'Total time: {end - start}')
