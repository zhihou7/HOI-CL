#!/bin/bash

echo "Setting up symlinks for precomputed hoi detection..."

dir_name=( "union" "ho" "ho_0" "ho_1" "ho_s" "ho_1_s" )

cd output

for k in "${dir_name[@]}"; do
  if [ -L $k ]; then
    rm $k
  fi
  if [ -d $k ]; then
    echo "Failed: ouput/$k already exists as a folder..."
    continue
  fi
  ln -s precomputed_hoi_detection/$k $k
done

cd ..

echo "Done."
