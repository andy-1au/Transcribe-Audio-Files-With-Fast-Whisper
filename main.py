from faster_whisper import WhisperModel

# Import the model
model_size = "large-v2"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, _ = model.transcribe("audiomass-output.mp3")

# Create a list to store the transcription segments
transcription_text = []

for segment in segments:
    transcription_text.append(segment.text)

# Join the segments into a single text, with newlines per each timestamp
full_transcription = ''.join(transcription_text)

# Define the file name for the output text file
output_file = "transcription.txt"

# Write the full transcription to the text file
with open(output_file, 'w') as file:
    file.write(full_transcription)

# Print a message indicating where the transcription was saved
print(f"Transcription saved to '{output_file}'")


