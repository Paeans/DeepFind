#!/bin/bash

zcat tmp.gz                                               #view gz files (line by line)


while read -r -a vararray; do command; done < file.txt    #read file.txt line by line and split line in to vararray
${array[@]}                                               #get all the elements in the array
${array[index]}                                           #get indexed elements in the array
${!array[@]}                                              #get all the indexes of the array


grep -w "word"                                            #get the line match the specific word, no *word*
grep -v "word"                                            #invert mode, get the line don't contain the word


