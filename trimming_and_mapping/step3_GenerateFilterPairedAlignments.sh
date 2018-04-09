#!/bin/bash
#$ -cwd
#$ -l mfree=8G
#s -l disk_free=10G

#Aligned reads from read mates are combined here
#We do not use resulting sam files but filtered .mapped files
#infile1 is aligned .mapped file for read1 from paired reads (R1)
#and infile2 is the same file from read2
#Outfile is the resulting sam-like file that contains pairs of aligned Hi-C reads

infile1=$1
infile2=$2
outfile=$3

# pairing of each alignment
join $infile1 $infile2 > $outfile


#PCR duplicates are cleaned here
infile=$3
out1=${infile%.*}
ext=${infile##*.}
#noPCR file contains paired interactions that are filtered from PCR duplicates
out1=$out1.noPCR.$ext

#infile=$1
#out1=$2


script1="script1.awk"
echo ' BEGIN {} ' > $script1
echo " {s1=\"+\"; s2=\"+\"; if (\$2==\"16\") {s1=\"-\"}; if (\$6==\"16\") {s2=\"-\"}; printf(\"%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n\", \$3, \$4, s1, \$7, \$8, s2, \$5, \$9, \$1)} " >> $script1
echo 'END {} ' >> $script1

script2="script2.awk"
echo ' BEGIN {} ' > $script2
echo "{ print \$9 \"\\t\" \$3 \"\\t\" \$1 \"\\t\" \$2 \"\\t\" \$7 \"\\t\" \$6 \"\\t\" \$4 \"\\t\" \$5 \"\\t\" \$8}" >> $script2
echo ' END {} ' >> $script2


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IN THE FOLLOWING LINE, MAKE SURE THE PATH TO sort_loci is update based on where you place these scripts
# CURRENTLY IT WILL ONLY WORK IF YOU CALL THE SCRIPTS WITHIN THE FOLDER WHERE THEY RESIDE
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python ./sort_loci.py $infile 2 | awk -f $script1 | sort -u -k1,6 | awk -f $script2 > $out1

rm $script1
rm $script2 




