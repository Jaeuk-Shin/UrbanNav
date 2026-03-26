#!/bin/bash

# Check if a file was provided
if [ -z "$1" ]; then
    echo "Usage: ./download_playlists.sh list.txt"
    exit 1
fi

URL_FILE=$1

# Check if the file exists
if [ ! -f "$URL_FILE" ]; then
    echo "Error: File '$URL_FILE' not found."
    exit 1
fi

# Loop through each line in the text file
while IFS= read -r url || [ -n "$url" ]; do
    # Skip empty lines and lines starting with #
    [[ -z "$url" || "$url" =~ ^# ]] && continue

    echo "------------------------------------------"
    echo "Processing: $url"
    echo "------------------------------------------"

    # Download command
    # -o creates a folder named after the playlist and titles the files properly
    # --yes-playlist ensures it treats the link as a playlist
    yt-dlp -f "bestvideo" \
       -o "%(playlist_title)s/%(playlist_index)s - %(title)s.%(ext)s" \
       --yes-playlist \
       --cookies-from-browser safari \
       --continue \
       --ignore-errors \
       "$url"

done < "$URL_FILE"

echo "All downloads complete!"
