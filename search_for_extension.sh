#!/bin/bash

file_path=''
file_extension_mp3='.mp3'
file_extension_mp4='.mp4'

find "$file_path" -type f \( -name "*$file_extension_mp3" -o -name "*$file_extension_mp4" \) > "$file_path.txt"

echo "File paths with $file_extension_mp3 or $file_extension_mp4 extension have been saved to $file_path.txt"
