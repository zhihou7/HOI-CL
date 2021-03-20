#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../output" && pwd )"
cd $DIR

FILE=precomputed_hoi_detection.tar.gz
ID=1B0AjsYQhXfFmOWDAuOwLEBSaP6BsoYah

if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading precomputed HOI detection (2.0G)..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
