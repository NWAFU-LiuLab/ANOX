# feature_extraction.py

This script extracts 1673 features from Data folder.

## Instruction manual

1. Decompress related_files(Part 1).rar and related_files(Part 2).rar.

2. Find `**Data**` folder in the unzipped folder. Then copy `**Data**` folder to the current folder.

3. Run Python script.

## Script introduction

* Feature extraction is based on five feature extraction methods ( FRE, AADP, EEDP, KSB and PRED).

* Script input: the information from Data folder.

* Script output: x_1673.txt and y.txt ( If the files already exist, they will be overwritten. )

* The output file can be used for feature selection in the second step.
