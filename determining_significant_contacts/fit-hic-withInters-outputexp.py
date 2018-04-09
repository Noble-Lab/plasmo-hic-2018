#!/usr/bin/env python
'''
Created on June 11,2012

@author: ferhat
'''

import sys
import os
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy import *
from scipy.interpolate import Rbf, UnivariateSpline
from scipy import optimize
from optparse import OptionParser
import scipy.special as scsp
import bisect
# R dependencies
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
fdrtool = importr('fdrtool')

from pylab import *

#### matplotlib fontsize settings
plt.rcParams['font.size']=17
plt.rcParams['axes.labelsize']='x-large'
plt.rcParams['xtick.labelsize']='large'
plt.rcParams['ytick.labelsize']='large'
plt.rcParams['figure.subplot.hspace']=0.4
plt.rcParams['figure.subplot.bottom']=0.12
plt.rcParams['figure.subplot.left']=0.15
plt.rcParams['figure.subplot.right']=0.94
plt.rcParams['figure.subplot.top']=0.92


# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

# global variables shared by functions
chrList=[] # list of all chromosomes
listOfMappableFrags=[] # list of all mappable fragments
# dictkey should be chr-locus1-locus2 and the return values are [interactionDistance, interactionCount]
possiblePairsPerDistance={} # dictionary of possible intra-chr pairs

# these are going to be the important numbers that we compute
possibleInterAllCount=0
observedInterAllCount=0
observedInterAllSum=0
# intra chromosomal interactions in range
possibleIntraInRangeCount=0
observedIntraInRangeCount=0
observedIntraInRangeSum=0
# all possible intra chromosomal interactions
possibleIntraAllCount=0
observedIntraAllCount=0
observedIntraAllSum=0

minGenomicDist=500000000 # some number bigger than the biggest chromosome
maxGenomicDist=0 

# normalized the inter-chr contanct probability so that it is not punished 
# just because there are so many mor pairs that are inter than intra
# it is now equivalent to 1.0/possibleIntraAllCount insted of 1.0/possibleInterAllCount
normalizedInterChrProb=0	# 1.0/possibleIntraAllCount
interChrProb=0	# 1.0/possibleInterAllCount

#This just avoids overflow - but is necessary for large genomes
distScaling=10000.0 

# DEFAULT values for program options 
noOfBins=100 # 100 by default 
distUpThres=-1 # -1 by default, means no upper bound
distLowThres=-1 # -1 by default, means no lower bound
mappabilityThreshold=1
useBinning=False
isSorted=False
libname=""

### Parameters that can be played with for the SPLINE FIT
residualFactor=1 # how many times better residual we want compared to power-law fit
overSample=5 # can be changed to have more/less overfitted splines
####

def main():
	### parse the command line arguments
	usage = "usage: %prog [options]"
	parser = OptionParser(usage=usage)
	parser.add_option("-f", "--fragments", dest="fragsfile",
					  help="midpoints of the fragments are read from FRAGSFILE")
	parser.add_option("-i", "--interactions", dest="intersfile",
					  help="interactions between fragment pairs are read from INTERSFILE")
	parser.add_option("-t", "--biases", dest="biasfile",
					  help="OPTIONAL: biases calculated by ICE for each locus are read from BIASFILE")
	parser.add_option("-r", "--residualFactor", dest="residualFactor", type="float",
					  help="OPTIONAL: how many times better residual is desired for splines compared to power-law fit. Default is 10")
	parser.add_option("-p", "--passes", dest="noOfPasses",type="int",
					  help="number of passes after the initial (before) fit. DEFAULT is 1 (after)")
	parser.add_option("-b", "--noOfBins", dest="noOfBins", type="int",
					  help="number of equal-occupancy (count) bins. Default is 100")
	parser.add_option("-m", "--mappabilityThres", dest="mappabilityThreshold", type="int",
					  help="minimum number of hits per locus that has to exist to call it mappable")
	parser.add_option("-l", "--lib", dest="libname",
					  help="Name of the library that is analyzed. Optional.")
	parser.add_option("-U", "--upperbound", dest="distUpThres", type="int",
					  help="upper bound on the intra-chromosomal distance range")
	parser.add_option("-L", "--lowerbound", dest="distLowThres", type="int",
					  help="lower bound on the intra-chromosomal distance range")
	parser.add_option("-n", "--nobinning",
					  action="store_false", dest="useBinning", help="do not use binning (default)")
	parser.add_option("-y", "--usebinning",
					  action="store_true", dest="useBinning", help="use binning")
	parser.add_option("-s", "--sorted",
					  action="store_true", dest="isSorted", help="NOT IMPLEMENTED YET - set this flag if interaction counts file is already sorted by genomic distance")
	parser.add_option("-v", "--verbose",
					  action="store_true", dest="verbose")
	parser.add_option("-q", "--quiet",
					  action="store_false", dest="verbose")
	parser.set_defaults(verbose=True, useBinning=True, noOfBins=100, distLowThres=-1, distUpThres=-1, mappabilityThreshold=1,isSorted=False,noOfPasses=1,residualFactor=10.0, biasfile='none')
	(options, args) = parser.parse_args()
	if len(args) != 0:
		parser.error("incorrect number of arguments")

	global noOfBins
	global distUpThres
	global distLowThres
	global useBinning
	global libname
	global mappabilityThreshold
	global isSorted
	global noOfPasses
	global residualFactor
	noOfBins=options.noOfBins # 100 by default 
	distUpThres=options.distUpThres # -1 by default, means no upper bound
	distLowThres=options.distLowThres # -1 by default, means no lower bound
	mappabilityThreshold=options.mappabilityThreshold # 1 by default
	useBinning=options.useBinning
	isSorted=options.isSorted
	libname=options.libname
	noOfPasses=options.noOfPasses
	residualFactor=options.residualFactor
	### call the functions ###

	# read the mandatory input files -f and -i
	generate_FragPairs(options.fragsfile)
	sortedInteractions=read_All_Interactions(options.intersfile)
	biasDic={}
	if options.biasfile!='none':
		biasDic=read_ICE_biases(options.biasfile)
	print >> sys.stderr, biasDic
	
	### DO THE FIRST PASS ###
	# calculate priors using original fit-hic and plot with standard errors
	x,y,yerr=calculate_Probabilities(sortedInteractions,[0 for i in range(len(sortedInteractions))],libname+".fithic_pass1")
	# first fit power-law to calculate the residual and plot the fit
	# now it also calculates the pvalues with power-law fit
	plawX,plawY,plawResidual,isOutlier=fit_Powerlaw(x,y,yerr,options.intersfile,sortedInteractions,biasDic,libname+".plaw_pass1")
	# now fit spline to the data using power-law residual by improving it  <residualFactor> times
	splineX,splineY,splineResidual,isOutlier=fit_Spline(x,y,yerr,options.intersfile,sortedInteractions,plawResidual,biasDic,libname+".spline_pass1")

	for i in range(2,2+noOfPasses):
		x,y,yerr=calculate_Probabilities(sortedInteractions,isOutlier,libname+".fithic_pass"+repr(i))
		plawX,plawY,plawResidual,isOutlier=fit_Powerlaw(x,y,yerr,options.intersfile,sortedInteractions,biasDic,libname+".plaw_pass"+repr(i))
		splineX,splineY,splineResidual,isOutlier=fit_Spline(x,y,yerr,options.intersfile,sortedInteractions,plawResidual,biasDic,libname+".spline_pass"+repr(i))

	return

	if False:

		### PLOT THE SPLINES AND POWER-LAWS BEFORE AND AFTER OUTLIER REMOVALL ###
		splines_Together(splineX,splineY,splineXNew,splineYNew,libname+".splines_compared")
		powerlaws_Together(plawX,plawY,plawXNew,plawYNew,libname+".plaws_compared")

		compare_with_Discrete_Binning(splineX,splineY,discreteBinsize,libname+".splineBefore_binning")
		compare_with_Discrete_Binning(splineXNew,splineYNew,discreteBinsize,libname+".splineAfter_binning")
			
	return # from main
	
def read_ICE_biases(infilename):
	biasDic={}
	infile =open(infilename, 'r')
	for line in infile:
		words=line.rstrip().split()
		chr=words[0]; midPoint=int(words[1]); bias=float(words[2])
		if chr not in biasDic:
			biasDic[chr]={}
		if midPoint not in biasDic[chr]:
			biasDic[chr][midPoint]=bias
	infile.close()
	return biasDic # from read_ICE_biases

def fit_Spline(x,y,yerr,infilename,sortedInteractions,plawResidual,biasDic,figname):
	sys.stderr.write("\nFit a univariate spline to the probability means\n")
	sys.stderr.write("------------------------------------------------------------------------------------\n")

	sys.stderr.write("interChromosomalProbability\t" + repr(interChrProb)+"\tnormalizedInterChromosomalProbability\t" + repr(normalizedInterChrProb)+"\n")
	# xi and yi will be used only for visualization purposes
	# acutal fit and residual is all done on vectors x and y
	xi = np.linspace(min(x), max(x), overSample*len(x))

	# use the power-law residual and set the desired error for the spline 
	# by dividing it to some factor of improvement that we want

	if residualFactor==-1: # this means use 
		splineError=min(y)*min(y)
	else:
		splineError=plawResidual/residualFactor

	# use fitpack2 method
	ius = UnivariateSpline(x, y, s=splineError)
	yi = ius(xi)

	#### POST-PROCESS THE SPLINE TO MAKE SURE IT'S NON-INCREASING
	### NOW I DO THIS BY CALLING AN R function CALLED MONOREG 
	### This does the isotonic regression using option antitonic to make sure 
	### I get monotonically decreasing probabilites with increasion genomic distance 
	splineX=sorted(list(set([int(i[0]) for i in sortedInteractions])))
	splineY=ius(splineX)
	# R vector format
	rSplineX=ro.FloatVector(splineX)
	rSplineY=ro.FloatVector(splineY)
	rMonoReg=ro.r['monoreg']
	# do the antitonic regression
	allRres=rMonoReg(rSplineX,rSplineY,type="antitonic")
	rNewSplineY=allRres[3]
	# convert data back to Python format
	newSplineY=[]
	diff=[]
	diffX=[]
	for i in range(len(rNewSplineY)):
		newSplineY.append(rNewSplineY[i])
		if (splineY[i]-newSplineY[i]) > 0:
			diff.append(splineY[i]-newSplineY[i])
			diffX.append(splineX[i])
	# END for
	
	# save the full spline
	np.savetxt(figname+"_spline.txt",newSplineY)
		
	### Now plot the results
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)
	residual =sum([i*i for i in (y - ius(x))])
	#plt.title('Univariate spline fit to the output of equal occupancy binning. \n Residual= %e' % (residual),size='small')
	#plt.plot([i/1000.0 for i in x], [i*100000 for i in y], 'ro', label="Means")
	#plt.plot([i/1000.0 for i in xi], [i*100000 for i in yi],'g-',label="Spline fit")
	plt.plot([i/1000.0 for i in splineX], [i*100000 for i in newSplineY],'g-',label="Spline fit")
	plt.errorbar([i/1000.0 for i in x],[i*100000 for i in y],yerr=[i*100000 for i in yerr], fmt='ro', label="Means")
	#plt.plot([i/1000.0 for i in x], [normalizedInterChrProb*100000 for i in x],'k-',label="Random intra-chromosomal")
	#plt.plot([i/1000.0 for i in x], [interChrProb*100000 for i in x],'b-',label="Inter-chromosomal")
	plt.ylabel('Contact probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	#plt.xlim([min(x)/1000.0,max(x)/1000.0])
	plt.xlim([20,1000])
	plt.ylim([0,6])

	ax.legend(loc="upper right")

	#ax = fig.add_subplot(2,1,2)
	#plt.loglog(splineX,newSplineY,'g-')
	##plt.loglog(xi, yi, 'g-') 
	#plt.loglog(x, y, 'r.')  # Data
	#plt.loglog(x, [normalizedInterChrProb for i in x],'k-')
	#plt.loglog(x, [interChrProb for i in x],'b-')
	#plt.ylabel('Probability (log scale)')
	#plt.xlabel('Genomic distance (log scale)')
	##plt.xlim([20000,100000])
	#plt.xlim([min(x),max(x)])
	plt.savefig(figname+'.png')
	plt.savefig(figname+'.pdf')
	sys.stderr.write("Plotting %s" % figname + ".png\n")

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.title('The amount of probabilities that needed to be subtracted in order to\nensure our spline is monotonically non-increasing.',size='small')
	plt.plot([i/1000.0 for i in diffX], [i*100000 for i in diff], 'r.', label="Amount subtracted")
	plt.ylabel('Probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	ax.legend(loc="upper right")
	plt.savefig(figname+'_monotonicRegDiff.png')
	sys.stderr.write("Plotting %s" % figname + "_monotonicRegDiff.png\n")

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.plot([i/1000.0 for i in splineX], [max(i*100000,0) for i in splineY], 'g.-',label="Before monotonic regression")
	plt.plot([i/1000.0 for i in splineX], [max(i*100000,0) for i in newSplineY], 'r-',label="After monotonic regression")
	#plt.plot([i/1000.0 for i in x], [i*100000 for i in y], 'ro', label="Means")
	plt.title('Splines before and after the post-processing step that ensures monotonicity.',size='small')
	plt.ylabel('Probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	#plt.xlim([20,50])
	ax.legend(loc="upper right")
	plt.savefig(figname+'_monotonicReg.png')
	sys.stderr.write("Plotting %s" % figname + "_monotonicReg.png\n")

	# NOW write the calculated pvalues and corrected pvalues in a file 
	outfile =open(figname+'.pvals.txt', 'w')
	infile =open(infilename, 'r')
	intraInRangeCount=0
	intraOutOfRangeCount=0
	intraVeryProximalCount=0
	interCount=0
	sys.stderr.write("distLowThres " + repr(distLowThres) + "\tdistUpThres " + repr(distUpThres) +"\n")
	p_vals=[]
	q_vals=[]
	expected_vals = []
	expected_vals_norm = []
	for line in infile:
		words=line.rstrip().split()
		chrNo1=words[0]
		midPoint1=int(words[1])
		chrNo2=words[2]
		midPoint2=int(words[3])
		interactionCount=float(words[4])
		bias1=1.0; bias2=1.0;  # assumes there is no bias to begin with
		# if the biasDic is not null sets the real bias values
		if len(biasDic)>0:
			bias1=biasDic[chrNo1][midPoint1]
			bias2=biasDic[chrNo2][midPoint2]

		if chrNo1==chrNo2: # if intra
			interactionDistance=abs(midPoint1-midPoint2)
			if (distLowThres==-1 or (distLowThres>-1 and interactionDistance > distLowThres)) and\
				(distUpThres==-1 or (distUpThres>-1 and interactionDistance <= distUpThres)): # intra and in range

				# make sure the interaction distance is covered by the probability bins
				distToLookUp=max(interactionDistance,min(x))
				distToLookUp=min(distToLookUp,max(x))
				i=min(bisect.bisect_left(splineX, distToLookUp),len(splineX)-1) 
				prior_p=newSplineY[i]*(bias1*bias2) # biases added in the picture
				intraInRangeCount +=1
				############# THIS HAS TO BE interactionCount-1 ##################
				p_val=scsp.bdtrc(interactionCount-1,observedIntraInRangeSum,prior_p)
				e_val = observedIntraInRangeSum*prior_p
				e_val_norm = e_val/(bias1*bias2)
			elif (distLowThres>-1 and interactionDistance <= distLowThres): # eliminate these very proximal interactions by setting prior to 1
				prior_p=1.0
				p_val=1.0
				e_val = observedIntraInRangeSum
				e_val_norm = e_val
				intraVeryProximalCount +=1
			else: # out of range bigger than distUpThres - treat it as inter chromosomal
				prior_p=normalizedIntraChrProb*(bias1*bias2) # biases added in the picture
				############# THIS HAS TO BE interactionCount-1 ##################
				p_val=scsp.bdtrc(interactionCount-1,observedInterAllSum,prior_p)
				e_val=observedInterAllSum*prior_p
				e_val_norm = e_val/(bias1*bias2)
				intraOutOfRangeCount +=1
			# END if
		else:
			#prior_p=normalizedInterChrProb
			prior_p=interChrProb*(bias1*bias2) # biases added in the picture
			############# THIS HAS TO BE interactionCount-1 ##################
			p_val=scsp.bdtrc(interactionCount-1,observedInterAllSum,prior_p)
			e_val =observedInterAllSum*prior_p
			e_val_norm = e_val/(bias1*bias2)
			interCount +=1
		p_vals.append(p_val)
		expected_vals.append(e_val)
		expected_vals_norm.append(e_val_norm)
	# END for
	infile.close()

	# Do the BH FDR correction 
	q_vals=benjamini_hochberg_correction(p_vals, possibleInterAllCount+possibleIntraAllCount)
	sys.stderr.write("possibleInterAllCount+possibleIntraAllCount " + repr(possibleInterAllCount+possibleIntraAllCount)+"\n")
	infile =open(infilename, 'r')
	count=0
	for line in infile:
		words=line.rstrip().split()
		chrNo1=words[0]
		midPoint1=int(words[1])
		chrNo2=words[2]
		midPoint2=int(words[3])
		interactionCount=float(words[4])
		p_val=p_vals[count]
		q_val=q_vals[count]
		e_val=expected_vals[count]
		e_val_norm = expected_vals_norm[count]
		count+=1
		outfile.write("%s\t%d\t%s\t%d\t%.2f\t%e\t%e\t%e\t%e\n" % (str(chrNo1),midPoint1,str(chrNo2),midPoint2,\
			interactionCount,p_val,q_val,e_val,e_val_norm))
	# END for - printing pvals and qvals for all the interactions

	isOutlier=[]
	distsBelow=[]
	distsAbove=[]
	intcountsBelow=[]
	intcountsAbove=[]
	belowThresCount=0
	aboveThresCount=0
	outlierThres=1.0/possibleIntraAllCount
	for interactionDistance,interactionCount in sortedInteractions:
		# make sure the interaction distance is covered by the probability bins
		distToLookUp=max(interactionDistance,min(x))
		distToLookUp=min(distToLookUp,max(x))
		i=min(bisect.bisect_left(splineX, distToLookUp),len(splineX)-1) 
		prior_p=newSplineY[i]
		############# THIS HAS TO BE interactionCount-1 ##################
		p_val=scsp.bdtrc(interactionCount-1,observedIntraInRangeSum,prior_p)
		if p_val < outlierThres:
			distsBelow.append(interactionDistance)
			intcountsBelow.append(interactionCount)
			isOutlier.append(1)
			belowThresCount +=1
		else:
			distsAbove.append(interactionDistance)
			intcountsAbove.append(interactionCount)
			isOutlier.append(0)
			aboveThresCount +=1
	# END for - doing the outlier check for all interactions in sortedInteractions

	sys.stderr.write("Plotting results of extracting outliers to file %s" % figname + ".extractOutliers.png\n")
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#plt.plot(dists, pvals, 'b.')
	plt.plot([i/1000.0 for i in distsBelow], intcountsBelow, 'g.',label="Pairs with p-value < 1/M (outliers)")
	plt.plot([i/1000.0 for i in distsAbove], intcountsAbove, 'r.',label="Pairs with p-value >= 1/M (part of null)")
	plt.xlabel('Genomic distance (kb)')
	plt.ylabel('Interaction count')
	#plt.xlim([distLowThres,distUpThres])
	#plt.ylim([0,max(max(intcounts),max(intcountsAbove))])
	#plt.plot(x, y, 'g')
	plt.title('Pairs below 1/M threshold: '+str(belowThresCount)+' -- '+'above: '+str(aboveThresCount))
	ax.legend(loc="upper right",fancybox=True)
	plt.savefig(figname+'.extractOutliers.png')

	sys.stderr.write("intraInRangeCount " + repr(intraInRangeCount)+"\tintraOutOfRangeCount " +\
		repr(intraOutOfRangeCount)+"\tintraVeryProximalCount " + repr(intraVeryProximalCount) +"\tinterCount " + repr(interCount)+"\n")

	sys.stderr.write("Writing p-values to file %s" % figname + ".txt\n")

	sys.stderr.write("Plotting q-values to file %s" % figname + ".qplot.png\n")
	minFDR=0.0
	maxFDR=0.20
	increment=0.005
	plot_qvalues(q_vals,minFDR,maxFDR,increment,figname+".qplot")

	infile.close()
	outfile.close()

	return [splineX, newSplineY, residual, isOutlier] # from fit_Spline


def fit_Powerlaw(xdata,ydata,yerr,infilename,sortedInteractions,biasDic,figname):
	sys.stderr.write("\nFit the optimal power-law to the probability means and standard deviations\n")
	sys.stderr.write("Biases are not considered yet. TODO\n")
	sys.stderr.write("------------------------------------------------------------------------------------\n")
	logx =log10(xdata)
	logy =log10(ydata)
	logyerr =log10(yerr)

	# define our (line) fitting function
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=1)
	pfinal = out[0]
	covar = out[1]
	index = pfinal[1]
	amp = 10.0**pfinal[0]
	indexErr = sqrt( covar[0][0] )
	ampErr = sqrt( covar[1][1] ) * amp
	sys.stderr.write('Ampli = %5.2f +/- %5.2f\n' % (amp, ampErr)) 
	sys.stderr.write('Index = %5.2f +/- %5.2f\n' % (index, indexErr)) 

	# calculate the probability for inter chromosomal interactions
	# simply 1 divided by the number of all possible  inter-chr pairs
	sys.stderr.write("interChromosomalProbability\t" + repr(interChrProb)+"\tnormalizedInterChromosomalProbability\t" + repr(normalizedInterChrProb)+"\n")
	# plot the figure 
	residual=0
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)
	#plt.plot(xdata, powerlaw(xdata, amp, index))	 # Fit
	#plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.') # Data
	residual =sum([i*i for i in (ydata - powerlaw(xdata, amp, index))])
	plt.title('Best power-law fit to the output of equal occupancy binning.\n Residual= %e' % (residual),size='small')
	#plt.plot([i/1000.0 for i in xdata], [i*100000 for i in ydata], 'ro', label="Mean")
	plt.errorbar([i/1000.0 for i in xdata], [i*100000 for i in ydata], [i*100000 for i in yerr], fmt='r.', label="Mean with std. error")
	plt.plot([i/1000.0 for i in xdata], [i*100000 for i in powerlaw(xdata, amp, index)],'g-',label="Power-law fit")
	plt.plot([i/1000.0 for i in xdata], [normalizedInterChrProb*100000 for i in xdata],'k-',label="Random intra-chromosomal")
	plt.plot([i/1000.0 for i in xdata], [interChrProb*100000 for i in xdata],'b-',label="Inter-chromosomal")
	plt.ylabel('Probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	ax.legend(loc="upper right")
	ax = fig.add_subplot(2,1,2)
	plt.loglog(xdata, powerlaw(xdata, amp, index),'g-')
	plt.errorbar(xdata, ydata, yerr=yerr, fmt='r.') # Data
	plt.loglog(xdata, [normalizedInterChrProb for i in xdata],'k-')
	plt.loglog(xdata, [interChrProb for i in xdata],'b-')
	#plt.ylim([pow(10,-9), pow(10,-3)])
	plt.ylabel('Probability (log scale)')
	plt.xlabel('Genomic distance (log scale)')
	plt.savefig(figname+'.png')
	sys.stderr.write("Plotting %s" % figname + ".png\n")

	outfile =open(figname+'.pvals.txt', 'w')
	infile =open(infilename, 'r')
	intraInRangeCount=0
	intraOutOfRangeCount=0
	intraVeryProximalCount=0
	interCount=0
	sys.stderr.write("distLowThres " + repr(distLowThres) + "\tdistUpThres " + repr(distUpThres) +"\n")
	p_vals=[]
	q_vals=[]
	for line in infile:
		words=line.rstrip().split()
		chrNo1=words[0]
		midPoint1=int(words[1])
		chrNo2=words[2]
		midPoint2=int(words[3])
		interactionCount=float(words[4])

		if chrNo1==chrNo2: # if intra
			interactionDistance=abs(midPoint1-midPoint2)
			if (distLowThres==-1 or (distLowThres>-1 and interactionDistance > distLowThres)) and\
				(distUpThres==-1 or (distUpThres>-1 and interactionDistance <= distUpThres)): # intra and in range
				# make sure the interaction distance is covered by the probability bins
				distToLookUp=max(interactionDistance,min(xdata))
				distToLookUp=min(distToLookUp,max(xdata))
				prior_p=powerlaw(distToLookUp,amp,index)
				intraInRangeCount +=1
				############# THIS HAS TO BE interactionCount-1 ##################
				p_val=scsp.bdtrc(interactionCount-1,observedIntraInRangeSum,prior_p)
			elif (distLowThres>-1 and interactionDistance <= distLowThres): # eliminate these very proximal interactions by setting prior to 1
				prior_p=1.0
				p_val=1.0
				intraVeryProximalCount +=1
			else: # out of range bigger than distUpThres - treat it as inter chromosomal
				prior_p=normalizedInterChrProb
				############# THIS HAS TO BE interactionCount-1 ##################
				p_val=scsp.bdtrc(interactionCount-1,observedInterAllSum,prior_p)
				intraOutOfRangeCount +=1
			# END if
		else:
			#prior_p=normalizedInterChrProb
			prior_p=interChrProb
			############# THIS HAS TO BE interactionCount-1 ##################
			p_val=scsp.bdtrc(interactionCount-1,observedInterAllSum,prior_p)
			interCount +=1
		p_vals.append(p_val)
	
	# END for
		# Bonferroni corrected qvalue
		#q_val=p_val*(possibleInterAllCount+possibleIntraAllCount)
		#outfile.write("%s\t%d\t%s\t%d\t%d\t%e\t%e\n" % (str(chrNo1),midPoint1,str(chrNo2),midPoint2,\
		#	interactionCount,p_val,q_val))
	infile.close()

	# Do the BH FDR correction 
	q_vals=benjamini_hochberg_correction(p_vals, possibleInterAllCount+possibleIntraAllCount)
	sys.stderr.write("possibleInterAllCount+possibleIntraAllCount " + repr(possibleInterAllCount+possibleIntraAllCount)+"\n")
	infile =open(infilename, 'r')
	count=0
	for line in infile:
		words=line.rstrip().split()
		chrNo1=words[0]
		midPoint1=int(words[1])
		chrNo2=words[2]
		midPoint2=int(words[3])
		interactionCount=float(words[4])
		p_val=p_vals[count]
		q_val=q_vals[count]
		count+=1
		outfile.write("%s\t%d\t%s\t%d\t%.2f\t%e\t%e\n" % (str(chrNo1),midPoint1,str(chrNo2),midPoint2,\
			interactionCount,p_val,q_val))
	# END for - printing pvals and qvals for all the interactions

	isOutlier=[]
	distsBelow=[]
	distsAbove=[]
	intcountsBelow=[]
	intcountsAbove=[]
	belowThresCount=0
	aboveThresCount=0
	outlierThres=1.0/possibleIntraAllCount
	for interactionDistance,interactionCount in sortedInteractions:
		prior_p=powerlaw(interactionDistance,amp,index)
		############# THIS HAS TO BE interactionCount-1 ##################
		p_val=scsp.bdtrc(interactionCount-1,observedIntraInRangeSum,prior_p)
		if p_val < outlierThres:
			distsBelow.append(interactionDistance)
			intcountsBelow.append(interactionCount)
			isOutlier.append(1)
			belowThresCount +=1
		else:
			distsAbove.append(interactionDistance)
			intcountsAbove.append(interactionCount)
			isOutlier.append(0)
			aboveThresCount +=1
	# END for - doing the outlier check for all interactions in sortedInteractions

	sys.stderr.write("Plotting results of extracting outliers to file %s" % figname + ".extractOutliers.png\n")
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#plt.plot(dists, pvals, 'b.')
	plt.plot([i/1000.0 for i in distsBelow], intcountsBelow, 'g.',label="Pairs with p-value < 1/M (outliers)")
	plt.plot([i/1000.0 for i in distsAbove], intcountsAbove, 'r.',label="Pairs with p-value >= 1/M (part of null)")
	plt.xlabel('Genomic distance (kb)')
	plt.ylabel('Interaction count')
	#plt.xlim([distLowThres,distUpThres])
	#plt.ylim([0,max(max(intcounts),max(intcountsAbove))])
	#plt.plot(x, y, 'g')
	plt.title('Pairs below 1/M threshold: '+str(belowThresCount)+' -- '+'above: '+str(aboveThresCount))
	ax.legend(loc="upper right",fancybox=True)
	plt.savefig(figname+'.extractOutliers.png')

	sys.stderr.write("intraInRangeCount " + repr(intraInRangeCount)+"\tintraOutOfRangeCount " +\
		repr(intraOutOfRangeCount)+"\tintraVeryProximalCount " + repr(intraVeryProximalCount) +"\tinterCount " + repr(interCount)+"\n")

	sys.stderr.write("Writing p-values to file %s" % figname + ".txt\n")

	sys.stderr.write("Plotting q-values to file %s" % figname + ".qplot.png\n")
	minFDR=0.0
	maxFDR=0.20
	increment=0.005
	plot_qvalues(q_vals,minFDR,maxFDR,increment,figname+".qplot")

	infile.close()
	outfile.close()

	return [xdata, powerlaw(xdata, amp, index), residual, isOutlier] # from fit_Powerlaw

def calculate_Probabilities(sortedInteractions,isOutlier,figname):

	sys.stderr.write("\nCalculating probability means and standard deviations by equal occupancy binning of interaction data\n")
	sys.stderr.write("------------------------------------------------------------------------------------\n")
	#sys.stderr.write("Plots the output in the file "+ figname + ".png\n")
	
	outfile=open(figname+'.txt', 'w')
	global noOfBins
	global observedIntraInRangeSum
	

	# total interaction count to put on top of the plot
	# this may be different than observedIntraInRangeSum for the second iteration of fit-hic
	totalInteractionCountForPlot=0
	lcount=0
	for eachrow in sortedInteractions:
		if isOutlier[lcount]==0:
			totalInteractionCountForPlot += eachrow[1]
		lcount+=1
	# END for
	desiredPerBin=(observedIntraInRangeSum)/noOfBins
	sys.stderr.write("observedIntraInRangeSum\t"+repr(observedIntraInRangeSum)+ "\tdesiredPerBin\t" +repr(desiredPerBin)+"\tnoOfBins\t"+repr(noOfBins)+"\n")

	# the following five lists will be the print outputs
	x=[] # avg genomic distances of bins
	y=[] # avg interaction probabilities of bins
	yerr=[] # stderrs of bins
	pairCounts=[] # number of pairs in bins
	interactionTotals=[] # number of interactions (reads) in bins

	# the following variables will be used to calculate the above five lists
	noOfPairsForBin=0
	meanCountPerPair=0
	M2=0
	interactionTotalForBin=0
	interactionTotalForBinTermination=0
	distanceTotalForBin=0
	lastDistanceForBin=-1
	lastInteraction=lcount 
	lcount=0 # this will increase by eachrow in sortedInteractions

	for eachrow in sortedInteractions:
		interactionDistance=eachrow[0]
		interactionCount=eachrow[1]
 
 		# if one bin is full or it's the last bin
		if noOfPairsForBin>0 and ((useBinning==False and lastDistanceForBin!=-1 and lastDistanceForBin!=interactionDistance) or\
			(useBinning==True and lastDistanceForBin!=-1 and interactionTotalForBinTermination >= desiredPerBin and\
			lastDistanceForBin!=interactionDistance) or lcount==lastInteraction): 

			# calculate the things that need to be calculated
			avgDistance=(distanceTotalForBin/noOfPairsForBin)*distScaling
			meanProbabilityObsv= (meanCountPerPair*1.0) /observedIntraInRangeSum
			se_p=meanProbabilityObsv
			# update se_p if there are more than 1 pairs in the bin
			if noOfPairsForBin>1:
				var=M2/(noOfPairsForBin-1)
				sd=math.sqrt(var)
				se=sd/math.sqrt(noOfPairsForBin)
				se_p=se/observedIntraInRangeSum
			# END if

			#sys.stderr.write(repr(avgDistance)+ "\t" +repr(meanProbabilityObsv)+"\t" +repr(se_p)+"\t" +repr(noOfPairsForBin) +"\t" +repr(interactionTotalForBin)+"\n")
			# append the calculated vals to corresponding lists
			x.append(float(avgDistance))
			y.append(float(meanProbabilityObsv))
			yerr.append(float(se_p))
			pairCounts.append(noOfPairsForBin)
			interactionTotals.append(interactionTotalForBin)
	
			# now that we saved what we need
			# set the values back to defaults and go on to the next bin
			noOfPairsForBin=0
			meanCountPerPair=0
			M2=0
			interactionTotalForBin=0
			interactionTotalForBinTermination=0
			distanceTotalForBin=0
			lastDistanceForBin=-1
		# END if - that checks whether the bin is full etc.

		# Now go back to processing the read values of interactionDistance and interactionCount
		# this check is necessary for the second pass of fit-hic
		# we want to only use the non-outlier interactions in our
		# probability calculation
		if isOutlier[lcount]==0:
			distanceTotalForBin +=interactionDistance/distScaling
			interactionTotalForBin +=interactionCount
			noOfPairsForBin +=1
			delta=interactionCount-meanCountPerPair
			meanCountPerPair += (delta*1.0) / noOfPairsForBin
			M2 +=delta*(interactionCount-meanCountPerPair)
		# END if
		interactionTotalForBinTermination +=interactionCount
		lcount +=1
		lastDistanceForBin=interactionDistance
	# END for over sortedInteractions

	#for i in range(len(x)):
	#	outfile.write("%d" % x[i] + "\t"+"%.2e" % y[i]+ "\t" + "%.2e" % yerr[i] + "\t" +"%d" % pairCounts[i] + "\t" +"%d" % interactionTotals[i]+"\n")
	#outfile.close()

	# first plot the figure before post-processing things
	sys.stderr.write("Plotting %s" % figname + ".png\n")
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot([i/1000.0 for i in x], [i*100000 for i in y],'ro',label="Mean")
	plt.errorbar([i/1000.0 for i in x], [i*100000 for i in y], [i*100000 for i in yerr], fmt='k.', label="Standard error")
	plt.ylabel('Probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	titleStr='Binning observed interactions using equal occupancy bins.\n No. of bins: '\
		+str(noOfBins) +', Library: ' + str(libname)+ ', No. of interactions: ' +str(observedIntraInRangeSum)
	plt.title(titleStr,size='small')
	ax.legend(loc="upper right")
	plt.savefig(figname+'.png')

	# I will do some post-processing now to make sure the probability values cover the whole x-range
	# I interpolate the gradient between the first 5 bins to minGenomicDist
	# NOW I'm skipping this part of postprocessing

	# print the post-processed x,y ... values
	for i in range(len(x)):
		outfile.write("%d" % x[i] + "\t"+"%.2e" % y[i]+ "\t" + "%.2e" % yerr[i] + "\t" +"%d" % pairCounts[i] + "\t" +"%d" % interactionTotals[i]+"\n")
	outfile.close()
	return [x,y,yerr] # from calculate_Probabilities



def read_All_Interactions(infilename):
	sys.stderr.write("\nReading all the interactions and then sorting the intra chr ones according to genomic distance\n")
	sys.stderr.write("------------------------------------------------------------------------------------\n")

	global observedIntraAllSum
	global observedIntraAllCount
	global observedIntraInRangeSum
	global observedIntraInRangeCount
	global observedInterAllSum
	global observedInterAllCount
	global isSorted
	global minGenomicDist
	global maxGenomicDist

	#read the interactions file - create a two dimensional numpy array with each row is a [distance,count] pair
	infile =open(infilename, 'r')
	for line in infile:
		words=line.rstrip().split()
		chrNo1=words[0]
		midPoint1=int(words[1])
		chrNo2=words[2]
		midPoint2=int(words[3])
		interactionCount=float(words[4])
		chrIndex1=chrList.index(chrNo1)
		chrIndex2=chrList.index(chrNo2)

		if midPoint1 not in listOfMappableFrags[chrIndex1] or midPoint2 not in listOfMappableFrags[chrIndex2]:
			sys.stderr.write("Not-mappable fragment pair: %s\t" %str(chrNo1)+"%d\t" % midPoint1+ "%s\t" %str(chrNo2) +"%d\n" % midPoint2)
			continue

		if chrNo1==chrNo2: # intra-chromosomal
			interactionDistance=abs(midPoint1-midPoint2)
			if (distLowThres==-1 or (distLowThres>-1 and interactionDistance > distLowThres))\
				and (distUpThres==-1 or (distUpThres>-1 and interactionDistance <= distUpThres)):
				minGenomicDist=min(minGenomicDist,interactionDistance)
				maxGenomicDist=max(maxGenomicDist,interactionDistance)
				# every pair should already be in the dictionary with a zero interaction count
				dictkey=str(chrNo1)+'-'+str(min(midPoint1,midPoint2))+'-'+str(max(midPoint1,midPoint2))
				if not dictkey in possiblePairsPerDistance:
					sys.stderr.write("Illegal fragment pair: %s\t" %str(chrNo1)+"%d\t" % midPoint1+ "%s\t" %str(chrNo2) +"%d\n" % midPoint2)
					sys.exit("Illegal fragment pair")
				else:
					possiblePairsPerDistance[dictkey]=[interactionDistance,interactionCount]

				observedIntraInRangeSum +=interactionCount
				observedIntraInRangeCount +=1
			# END if
			observedIntraAllSum +=interactionCount
			observedIntraAllCount +=1
		else: # inter chromosomal
			observedInterAllSum +=interactionCount
			observedInterAllCount +=1
	# END for
	infile.close()
	sys.stderr.write("Total of \t"+str(observedIntraAllCount) +" observed intra-chr fragment pairs,\t"\
		+str(observedIntraInRangeCount) +" observed intra-chr fragment pairs in range,\t"\
		+str(observedInterAllCount) +" observed inter-chr fragment pairs\n" )
	sys.stderr.write("Total of \t"+str(observedIntraAllSum) +" observed intra-chr read counts,\t"\
		+str(observedIntraInRangeSum) +" observed intra-chr read counts in range,\t"\
		+str(observedInterAllSum) +" observed inter-chr read counts\n" )
	sys.stderr.write("Range of genomic distances	[%d	%d]" % (minGenomicDist,maxGenomicDist) + "\n")

	# sort the interactions if not already sorted
	sortedInteractions=[]
	for i in possiblePairsPerDistance:
		sortedInteractions.append(possiblePairsPerDistance.get(i))
	
	if isSorted==False:
		t=time.time()
		sortby_inplace(sortedInteractions,0)
		sys.stderr.write("Total time for sorting interactions according to genomic distance: %.3f\n" % (time.time()-t))
	else:
		sys.stderr.write("NOT IMPLEMENTED YET - Skipping sorting phase according to genomic distance since isSorted flag is set to TRUE\n")

	return sortedInteractions #from read_All_Interactions


def generate_FragPairs(infilename):
	sys.stderr.write("\nGenerating all possible intra-chromosomal fragment pairs and counting the number of all possible inter-chr fragment pairs\n")
	sys.stderr.write("------------------------------------------------------------------------------------\n")
	global listOfMappableFrags # two dimensional list with all mappable fragment midpoints for each chr
	global chrList # list of all chromosomes (chrno (type=int))
	global possiblePairsPerDistance # all possible intra-chr fragment pairs
	global possibleInterAllCount # count of all possible inter-chr fragment pairs
	global possibleIntraAllCount # count of all possible intra-chr fragment pairs
	global possibleIntraInRangeCount # count of all possible intra-chr fragment pairs in the range we're interested
	global interChrProb # 1 divided by all possible inter-chr fragment pairs 
	global normalizedInterChrProb #  1 divided by all possible intra-chr fragment pairs

	listOfMappableFrags=[]
	chrList=[]

	#get the name of the first chr
	infile =open(infilename, 'r')
	line=infile.readline()
	words=line.rstrip().split()
	currChrNo=words[0] #get the name of first chr
	infile.close()

	# read the fragments file 
	fragsPerChr=[] # temporary list that will be added to listOfMappableFrags for each chr
	totalNoOfFrags=0 # total number of all mappable fragments
	infile =open(infilename, 'r')
	for line in infile:
		words=line.rstrip().split()
		chrNo=words[0] # can be an integer or a string
		#words[1] ignored
		midPoint=int(words[2])
		hitCount=int(words[3])
		# whenever the name of the chromosome changes 
		if currChrNo!=chrNo:
			listOfMappableFrags.append(fragsPerChr)
			totalNoOfFrags += len(fragsPerChr)
			chrList.append(currChrNo)
			currChrNo = chrNo
			fragsPerChr=[]
		# add the mappable midPoints to the temp fragsPerChr
		if hitCount >= mappabilityThreshold:
			fragsPerChr.append(midPoint)
	#END for

	# handle the last chromosome
	listOfMappableFrags.append(fragsPerChr)
	totalNoOfFrags += len(fragsPerChr)
	chrList.append(currChrNo)
	infile.close()
	
	# create all possible frag pairs 
	possibleInterAllCount=0
	possibleIntraInRangeCount=0
	possibleIntraAllCount=0
	for i in chrList:
		countIntraPairs=0
		chrIndex=chrList.index(i) # get the index of chromosome from the chrList 
		fragsPerChr=(listOfMappableFrags[chrIndex])[:] # get the mappable midpoints for that chr
		tempLen=len(fragsPerChr)
		possibleInterAllCount+= (totalNoOfFrags-tempLen)*tempLen
		# iterate over all possible intra-chr pairs to see which ones qualify as a 'possible' pair
		for x in range(tempLen):
			for y in range(x+1,tempLen):
				interactionDistance=abs(fragsPerChr[x]-fragsPerChr[y])
				if (distLowThres==-1 or (distLowThres>-1 and interactionDistance >distLowThres)) and\
					(distUpThres==-1 or (distUpThres>-1 and interactionDistance <= distUpThres)):
					countIntraPairs +=1
					dictkey=str(i)+'-'+str(min(fragsPerChr[x],fragsPerChr[y]))+'-'+str(max(fragsPerChr[x],fragsPerChr[y]))
					possiblePairsPerDistance[dictkey]=[interactionDistance,0] # set count to zero for now
				possibleIntraAllCount+=1
			#END for
		#END for
		possibleIntraInRangeCount+=countIntraPairs
		sys.stderr.write("Chromosome " +repr(i) +",\t"+str(tempLen) +" mappable fragments, \t"+str(countIntraPairs)\
		+" possible intra-chr fragment pairs in range,\t" + str((totalNoOfFrags-tempLen)*tempLen) +" possible inter-chr fragment pairs\n")

	#END for

	# divide the possibleInterAllCount by 2 so that every inter-chr interaction is counted only once
	possibleInterAllCount=possibleInterAllCount/2
	sys.stderr.write("Total of \t"+str(possibleIntraInRangeCount) +" possible intra-chr fragment pairs in range,\t"\
	+str(possibleIntraAllCount) +" possible intra-chr fragment pairs,\t"\
	+str(possibleInterAllCount) +" possible inter-chr fragment pairs\n" )
	# calculate inter-chr probabilities
	if possibleInterAllCount>0:
		interChrProb=1.0/possibleInterAllCount
	else:
		interChrProb=0

	normalizedInterChrProb=1.0/possibleIntraAllCount

	return # from generate_FragPairs

def sortby_inplace(somelist, n):
	somelist[:] = [(x[n], x) for x in somelist]
	somelist.sort()
	somelist[:] = [val for (key, val) in somelist]
	return


def plot_qvalues(q_values,minFDR,maxFDR,increment,figname):
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	qvalTicks=np.arange(minFDR,maxFDR+increment,increment)
	significantTicks=[0 for i in range(len(qvalTicks))]
	qvalBins=[-1 for i in range(len(q_values))]
	for i, q in enumerate(q_values):
		if np.isnan(math.floor(q/increment)):
			qvalBins[i] = np.nan
		else:
			qvalBins[i]=int(math.floor(q/increment))
	
	for i in range(len(qvalBins)):
		if np.isnan(qvalBins[i]) or qvalBins[i]>=len(qvalTicks):
			continue
		significantTicks[qvalBins[i]]+=1
	
	# make it cumulative 
	for i in range(1,len(significantTicks)):
		significantTicks[i]=significantTicks[i]+significantTicks[i-1]

	plt.title('Number of significant locus pairs at varying FDR thresholds.', size='small')
	plt.plot(qvalTicks,significantTicks, 'b*-')
	plt.xlim([minFDR+increment,maxFDR])
	plt.xlabel('FDR threshold')
	plt.ylabel('Number of significant pairs')
	plt.savefig(figname+'.png')

	return

def benjamini_hochberg_correction(p_values, num_total_tests):
	# assumes that p_values vector might not be sorted
	pvalsArray=np.array(p_values)
	order=pvalsArray.argsort()
	sorted_pvals=np.take(p_values,order)
	q_values=[1.0 for i in range(len(p_values))]
	prev_bh_value = 0
	for i, p_value in enumerate(sorted_pvals):
		bh_value = p_value * num_total_tests / (i + 1)
		# Sometimes this correction can give values greater than 1,
		# so we set those values at 1
		bh_value = min(bh_value, 1)

		# To preserve monotonicity in the values, we take the
		# maximum of the previous value or this one, so that we
		# don't yield a value less than the previous.
		bh_value = max(bh_value, prev_bh_value)
		prev_bh_value = bh_value
		qIndex=order[i]
		q_values[qIndex] =bh_value
	#END for

	return q_values

if __name__ == "__main__":
	main()

