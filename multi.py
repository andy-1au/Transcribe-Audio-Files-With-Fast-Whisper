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
    file_name = os.path.basename(file_path)
    output_file = os.path.splitext(file_name)[0] + ".vtt"
    output_file_path = f'{fileConst.OUTPUT_FOLDER_PATH}/{output_file}'
    segments, info = model.transcribe(file_path, beam_size=5, vad_filter=True)
    count = 0
    with open(output_file_path, 'w') as file:
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

if __name__ == "__main__":
    start = time.time()

    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="int8", device_index=[0, 1, 2, 3])

    audio_paths_txt = fileConst.AUDIO_FILES_PATH
    audio_files = []
    try:
        with open(audio_paths_txt, 'r') as file:
            for line in file:
                path = line.strip()
                audio_files.append(path)
    except FileNotFoundError:
        print(f"File {audio_paths_txt} not found.")
    except IOError:
        print(f"Could not read file {audio_paths_txt}.")

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
