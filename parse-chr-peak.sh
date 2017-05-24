#!/bin/bash

PEAK_DATA_PATH=~/share/narrowPeak
HMGM_DATA_PATH=~/share/hg19
LABEL_DATA_PATH=~/share/label
DEEP_FIND=~/DeepFind

CHR_LENGTH_FILE=$DEEP_FIND/chr-length.txt
PEAK_FILE=$DEEP_FIND/gz-file-list.txt

seg_len=200

function init_chr_label {
  [[ "$1" == "" ]] || [[ "$2" == "" ]] && {
    echo "Function init_chr_label need two parameters"
    echo "param1: the name of the chr"
    echo "param2: the length of the chr"
    exit 1
  }
  chrfile=$LABEL_DATA_PATH/${1}-label.txt
  [[ -e $chrfile ]] && rm -f $chrfile
  
  seg_num=$(( chrlength / seg_len ))
  for i in `seq 0 $seg_num`; do
    echo -n -e $i\\t >> $chrfile
  done
  echo >> $chrfile
}

function parse_chr_file {
  while read -r -a chrinfo; do
    chrname=${chrinfo[0]}
    chrlength=`sed 's/\([0-9]\),\([0-9]\)/\1\2/g' <<< ${chrinfo[1]}`
    #seg_num=$(( chrlength / seg_len ))
    #rem_num=$(( chrlength % seg_len ))
    #echo $seg_num, $rem_num
    init_chr_label $chrname $chrlength
  done < $CHR_LENGTH_FILE ;
}

function calc_index {
  [[ "$1" == "" ]] || [[ "$2" == "" ]] && {
    echo "Function calc_index need two parameters"
    echo "param1: the start bp of a segment"
    echo "param2: the end bp of a segment"
    exit 1
  }
  startbp=$1
  endbp=$2
  
  sindex=$(( startbp / seg_len ))  
  rem_len=$(( startbp % seg_len ))  
  [[ $(( seg_len - rem_len )) -lt $(( seg_len / 2 )) ]] \
    && [[ $(( seg_len - rem_len )) -lt $(( (endbp - startbp)/2 )) ]] \
    && sindex=$(( sindex + 1 ))
  
  eindex=$(( endbp / seg_len ))
  rem_len=$(( endbp % seg_len ))
  [[ $rem_len -lt $(( seg_len / 2)) ]] \
    && [[ $rem_len -lt $(( (endbp - startbp)/2 )) ]] \
    && eindex=$(( eindex - 1 ))
  
  read -r -a indexlist <<< `seq $sindex $eindex`
  #echo ${indexlist[@]}
}

chrname=chr1
tfname=wgEncodeAwgTfbsUwWi38CtcfUniPk.narrowPeak.gz
chrlabelfile=$LABEL_DATA_PATH/${chrname}-label.txt

echo $tfname >> $chrlabelfile
zcat $PEAK_DATA_PATH/$tfname | grep -w "$chrname" | sort -k2 -n | while read -r -a chrinfo; do
  calc_index ${chrinfo[1]} ${chrinfo[2]}
  echo -n ${indexlist[@]}" " >> $chrlabelfile
done
echo >> $chrlabelfile
