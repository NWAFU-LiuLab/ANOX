# feature_selection.py

This script is used for feature selection, reducing 1673 features to 1170.

## Instruction manual

1. Decompress related_files(Part 1).rar and related_files(Part 2).rar.

2. Find ***x_1673.txt*** and ***y_txt*** in the unzipped folder. Then copy ***x_1673.txt*** and ***y_txt*** to the current folder

3. Run Python script

## Script introduction

* Feature selection is based on the Max-Relevance-Max-Distance (MRMD) algorithm.

* Script input: x_1673.txt and y.txt

* Script output: x_1170.txt ( If the file already exists, it will be overwritten. )

* The output file can be used for model validation in the third step.
