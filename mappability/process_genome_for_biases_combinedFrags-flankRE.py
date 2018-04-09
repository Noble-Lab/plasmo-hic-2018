#!/usr/bin/env python
'''
Created on Jun 2, 2012

@author: gunhan
@modified: ferhat 10/05/13

'''

import sys
import gzip
from datetime import datetime

USAGE = """

USAGE: process_genome_for_biases <fastaFileName> <fragsFileName> <howmanyFrags> <mappability files>

	<fastaFileName>		:	The fasta file which includes a SINGLE chromosome

	<fragsFileName>		:	The file with the fragments list. Expected to be in the form:
						<chr> <fragStart> <fragMid> <hitCount> <mappable?>

	<howmanyFrags>		:	How many frags are combined. NOT USED for now.

	<mappability files>	:	the mappability files which in gzipped plain text
							format. It includes one character (0-9)
							per base.

"""


def readFasta(filename):
	if filename.endswith(".gz"):
		fastafile=gzip.open(filename,'r')
	else:
		fastafile=open(filename,'r')
	#
	str_list=[]
	#read the fasta header
	fastaheader=fastafile.readline().rstrip()
	#read the rest of the file
	for line in fastafile:
		str_list.append(line.rstrip())
	chrName=fastaheader[1:]
	sequence=''.join(str_list)
	sequence=sequence.upper()
	return [chrName, sequence]

def readMappability(filename):
	mapFile=gzip.open(filename,'rb')
	str_list=[]
	#no fasta header, just read charcters
	for line in mapFile:
		str_list.append(line.rstrip())
	return ''.join(str_list)

def averageMappability(mappability):
	ln=len(mappability)
	if ln==0:
		return 0.0
	sum=0
	for i in mappability:
		k=ord(i)-32
		# set anything that is above 5 or zero to 5
		#if k==0 or k>5:
		#	k=5

		# mappability will be the proportion of uniquely mappable bases
		if k!=1:
			k=0

		sum=sum + k
	return float(sum)/float(ln)

def GCcontent(sequence):
	ln=0
	sum=0
	k={'G':1, 'C':2, 'g':3, 'c':4}
	n={'N':1, 'n':2}
	for i in sequence:
		if i in k:
			sum=sum+1
			ln=ln+1
		elif i not in n:
			ln=ln+1
	if ln==0:
		return 0
	else:
		return float(sum)/float(ln)

def main(fastaFile,fragsFile,howManyFrags,mapFile,REflank):
	[chrname, sequence]=readFasta(fastaFile)
	mappability=readMappability(mapFile)
	
	#now go bin by bin on the sequence, 
	start=0;
	prevStart=0

	infile =open(fragsFile, 'r')
	for line in infile:
		words=line.rstrip().split()
		if chrname!=words[0]:
			print "ERROR: chrname mismatch: "+ chrname+"\t vs "+words[0]
			exit
		if abs(start-int(words[2]))>2:
			print "ERROR: start index mismatch: "+str(start)+"\t vs "+str(words[1])
			exit

		# SWAPPED THE ORDER SO THAT fragmentMappability files are correct format
		mid=int(words[1])
		start=int(words[2])
		#
		if start >= len(sequence):
			break
		fragLen=2*(mid-start)
		afterFlankS=max(start-REflank,prevStart)
		afterFlankE=min(start+REflank,start+fragLen)

		avgmap=averageMappability(mappability[afterFlankS:afterFlankE])
		gccont=GCcontent(sequence[afterFlankS:afterFlankE])
		prevStart=start
		#output
		print("%s\t%d\t%d\t%.3f\t%.3f\n" % (chrname,start,fragLen+1,avgmap,gccont)),
		start=start+fragLen
		
	#
	infile.close()
	#print(str(datetime.now()))

if __name__ == "__main__":
	if (len(sys.argv) != 6):
		sys.stderr.write(USAGE)
		sys.exit(1)
	main(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]), str(sys.argv[4]), int(sys.argv[5]))

