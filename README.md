# DeepFind

Find regulation gene with deep learning model

Demo test base on dataset:
  
    EncodeDCC:  https://genome.ucsc.edu/ENCODE/downloads.html
    
    Roadmap:    https://personal.broadinstitute.org/anshul/projects/roadmap/peaks/consolidated/narrowPeak/



# **1. create_raw_label**

Files in create_raw_label directory used to generate peak name and corresponding segment id list

##  **1.1. Usage**
   
  ```
  
  parse_chr_peak.sh chr[1-22|X|Y] start_peak end_peak
    
  chr[1-22|X|Y]   : the name of chromesome to analysis

  start_peak      : from which peak in the gz-file-list.txt to start extract

  end_peak        : the end index of peak
  
  ```
  
##  **1.2. Result**
    
    peak_name   segment_1   segment_2   ... ...
    
    peak_name       : peak data file name (with gz extention)
    
    segment_*       : divide the gene sequence into segments of 200 bps, 
                      segment_* is the index of a peak on the chromesome
