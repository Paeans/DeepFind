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

-eq # equal
-ne # not equal
-lt # less than
-le # less than or equal
-gt # greater than
-ge # greater than or equal

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

myArray=('red' 'orange' 'green')

while false; do echo YES; done
while true; do echo YES; done
if true; then cmd; fi
if false; then cmd; fi

#!/bin/bash

ssh user@host "
  cd ~/share/label;
  for i in \`seq 0 22\`; do
    ./filename.sh chr8 \$i 40 &> /dev/null < /dev/null &  #use redirect, otherwise it will hang there
  done
  disown -a
"

for sname in `cat /etc/hosts | grep slave | awk '{print $2}'`; do ssh $sname "ps x | grep parse"; done
# if use while, will exit when one shell is finished

for sname in `cat /etc/hosts | grep slave | awk '{print $2}'`; do ssh $sname "echo ${password} | sudo -S command"; done
# sudo -S will read password from stdin, echo ${password} can give password to sudo command

function exec_remote {
for sname in `cat /etc/hosts | grep slave | awk '{print $2}'`;
do
  echo $sname;
  ssh $sname $1;
done
}

exec_remote "ls -l" # execute remote command using function style


for i in `seq 2 22`; do 
  sort -k1 chr$i-label-*.txt > chr$i-label.txt; 
  [[ -e raw-label/chr2-label ]] || mkdir raw-label/chr$i-label; 
  mv chr$i-label-*.txt ./raw-label/chr$i-label/; 
done


: << 'END'

code block need to comment
#used to comment a block of code
END

# create tunnel using ssh, then connect to remote host by through different port
ssh -i /home/pangaofeng/.ssh/id_rsa -f pangaofeng@localhost -L 0.0.0.0:200$i:slave0$i:22 -N

for i in `seq 1 11`;
do
  tmp=0${i}
  ssh -i /home/pangaofeng/.ssh/id_rsa -f pangaofeng@localhost -L 0.0.0.0:20${tmp: -2}:slave${tmp: -2}:22 -N
done

## for sname in `cat /etc/hosts | grep slave | awk '{print $2}'`;
## do
##   echo $sname ${sname: -2};
##   ssh -i /home/pangaofeng/.ssh/id_rsa -f pangaofeng@localhost -L 0.0.0.0:20${sname: -2}:${sname}:22 -N
## done


${tmp:+,${tmp},}
# add ',' at the begin and end of tmp

for i in `seq 1 20`;
do
  tmp=0${i}
  tmp=${tmp: -2}
  echo $tmp
done #generate a list of number have 2 digits, 
     #if not long enough, add 0 at the begin

for i in `for t in \`seq 1 5\`; do echo S$t;done`;do echo $i; done


# file format convert
#FASTQ to FASTA format:
zcat input_file.fastq.gz | awk 'NR%4==1{printf ">%s\n", substr($0,2)}NR%4==2{print}' > output_file.fa


#Illumina FASTQ 1.8 to 1.3
sed -e '4~4y/!"#$%&'\''()*+,-.\/0123456789:;<=>?@ABCDEFGHIJ/@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghi/' myfile.fastq   # add -i to save the result to the same input file


#Illumina FASTQ 1.3 to 1.8
sed -e '4~4y/@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghi/!"#$%&'\''()*+,-.\/0123456789:;<=>?@ABCDEFGHIJ/' myfile.fastq   # add -i to save the result to the same input file


#Illumina FASTQ 1.8 raw quality to binned quality (HiSeq Qtable 2.10.1, HiSeq 4000 )
sed -e '4~4y/!"#$%&'\''()*+,-.\/0123456789:;<=>?@ABCDEFGHIJKL/))))))))))----------77777<<<<<AAAAAFFFFFJJJJ/' myfile.fastq   # add -i to save the result to the same input file


#Illumina FASTQ 1.8 raw quality to clinto format (a visual block representation)
sed -e 'n;n;n;y/!"#$%&'\''()*+,-.\/0123456789:;<=>?@ABCDEFGHIJKL/▁▁▁▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇██████/' myfile.fastq   # add -i to save the result to the same input file

awk '($10 > 4) && ($10 < 10) {print}'


while true; do sensors coretemp-*; nvidia-smi; sleep 1; clear; done

#!/bin/bash

for filename in `ls *.bed`; do
echo $filename
mkdir ${filename}.dir
for cindex in `seq 1 22` X; do

echo chr${cindex}
awk -v chrname="chr${cindex}" '$1==chrname {print $1 " "$2 " "$3}' $filename > ${filename}.dir/${filename}_chr${cindex}.bed

done;
done;

matlab -nodisplay -nodesktop -nojvm -r 'addpath ..; run chr3' < /dev/null 2>&1 > /dev/null &
% run is the function defined in a .m file under the directories that included in matlab path
% addpath will add up level directory to the matlab path, then run is defined in a file at there

for i in `seq 1 12` X; 
do 
  nohup matlab -nodisplay -nodesktop -nojvm -r 'addpath ..; run chr'$i < /dev/null 2>&1 > /dev/null & 
done

latexdiff --packages=amsmath,endfloat,hyperref,apacite,siunitx,cleveref,glossaries,mhchem,chemformula old.tex new.tex > diff.tex
