#!/bin/bash
echo "argument1: $1"
mkdir -p cppData
n=1;
max=12;
while [ "$n" -le "$max" ]; do
  mkdir -p "cppData/subject$n"
  n=`expr "$n" + 1`;
done

echo "Successfully created the cpp folders"
touch signal.txt
echo $1 > signal.txt
cmake .
make
python3 ../generate_eeg.py
./eeg_filter
python3 ../get_results.py
python3 ../t_test.py
