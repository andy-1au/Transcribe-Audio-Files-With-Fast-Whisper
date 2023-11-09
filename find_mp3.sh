#!/bin/bash

# Get the root folder name
root_folder=drive_01_LU_29400 # CHANGE this

# Find MP3 files and save the list to a text file with the root folder name
find ${root_folder} -type f -name "*.mp3" > "${root_folder}_mp3_files_list.txt"
