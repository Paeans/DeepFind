#!/bin/bash

zcat tmp.gz                                               #view gz files (line by line)

while read -r -a vararray; do command; done < file.txt    #read file.txt line by line and split line in to vararray
${array[@]}                                               #get all the elements in the array
${array[index]}                                           #get indexed elements in the array
${!array[@]}                                              #get all the indexes of the array

grep -w "word"                                            #get the line match the specific word, no *word*
grep -v "word"                                            #invert mode, get the line don't contain the word

echo $?
|& <==> 2>&1

while read -r filename; do echo $filename; done < ../gz-file-list.txt | more
 
zcat filename.gz | grep -w "chr1" | sort -k2 -n
 
$PEAK_DATA_PATH
$HMGM_DATA_PATH
 
$(( 249250621 / 200 ))
$(( 249250621 % 200 ))

echo "1,123,345" | sed 's/\([0-9]\),\([0-9]\)/\1\2/g'     #remove the thousands separator from string
sed 's/\([0-9]\),\([0-9]\)/\1\2/g' <<< "1,234,567"

[[ -e filename ]]                                         #test filename exist or not
echo -n -e  abc\\tdef

echo ${str:startindex:exlen}

[[ $(( seg_len - rem_len )) == $(( seg_len / 2 )) ]]      #compare two string equal or not

[[ $(( seg_len - rem_len )) -lt $(( seg_len / 2 )) ]]     #compare two number big or small 
if (( ( seg_len - rem_len ) < ( seg_len / 2 ) )); then
  echo YES
fi                                                        #compare two number big or small

[[ test statement ]] && [[ test statement ]] && { statements }
[[ test statement ]] \
  && [[ test statement ]] \
  && { statements }
[[ test statement ]] &&
  [[ test statement ]] &&
  { statements }


function myfunc()
{
    local  __resultvar=$1
    local  myresult='some value'
    if [[ "$__resultvar" ]]; then
        eval $__resultvar="'$myresult'"
    else
        echo "$myresult"
    fi
}

myfunc result
echo $result
result2=$(myfunc)                                           #bash function to return value
echo $result2                                               #values can be set in the function and access outside function


