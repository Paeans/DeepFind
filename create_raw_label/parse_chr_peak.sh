#!/bin/bash
#
# ./parse-chr-peak.sh chrname peak_start_index peak_number

DEEP_FIND=`pwd`
SHARE_DIR=~/share
PEAK_DATA_PATH=$SHARE_DIR/narrowPeak
HMGM_DATA_PATH=$SHARE_DIR/hg19
LABEL_DATA_PATH=$SHARE_DIR/label


CHR_LENGTH_FILE=$DEEP_FIND/chr-length.txt
PEAK_FILE=$DEEP_FIND/gz-file-list.txt

seg_len=200

: <<'END'
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
END

function parse_chr_file {
  echo "Parse chrome list file"
  chr_name_list=()
  chr_len_list=()
  chr_info_index=0
  
  while read -r -a chrinfo; do
    chrname=${chrinfo[0]}
    chrlength=`sed 's/\([0-9]\),\([0-9]\)/\1\2/g' <<< ${chrinfo[1]}`
    
    chr_name_list[$chr_info_index]=$chrname
    chr_len_list[$chr_info_index]=$chrlength
    
    chr_info_index=$(( chr_info_index + 1 ))
  done < $CHR_LENGTH_FILE ;
}

function parse_peak_name {
  echo "Parse peak list file"
  peak_name_list=()
  peak_info_index=0
  
  while read -r peakname; do
    
    [[ "$1" == "" ]] || 
      [[ $(( $1 * $2 )) -le $peak_info_index ]] && #[[ $peak_info_index -lt $(( $1 * $2 + $2 )) ]] && 
      peak_name_list[$peak_info_index]=$peakname
    peak_info_index=$(( peak_info_index + 1 ))
    [[ "$2" != "" ]] && [[ $peak_info_index -ge $(( $1 * $2 + $2 )) ]] && break  
      # when not break from this line, outside of function will test as false
      # need at the end of function add return 0
  done < $PEAK_FILE ;
  return 0;
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

[[ "$1" != "" ]] && chr_name_list[0]="$1"  || 
parse_chr_file        && 
parse_peak_name $2 $3 &&
for chrname in ${chr_name_list[@]}; do
  chrlabelfile=$LABEL_DATA_PATH/${chrname}-label-"$2".txt
  [[ -e $chrlabelfile ]] && rm $chrlabelfile
  for tfname in ${peak_name_list[@]}; do
    echo -n $tfname" " >> $chrlabelfile
    zcat $PEAK_DATA_PATH/$tfname | grep -w "$chrname" | sort -k2 -n | {
      chr_seg_list=""
      while read -r -a chrinfo; do
        calc_index ${chrinfo[1]} ${chrinfo[2]}
        #echo -n ${indexlist[@]}" " >> $chrlabelfile
        chr_seg_list=$chr_seg_list${indexlist[@]}" "      
      done
      echo $chr_seg_list >> $chrlabelfile
    }
  done
done

