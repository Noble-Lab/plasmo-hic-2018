import numpy as np
import sys
from statsmodels.sandbox.stats.multicomp import multipletests

infile = sys.argv[1]
outfile = sys.argv[2]


data = np.loadtxt(infile)
positives = np.where(data>0) 
data[positives] = -data[positives]
pvals = np.exp(data)

# only process non-nan values
pvals_masked = np.ma.masked_invalid(pvals)
pvals_valid = pvals_masked[~pvals_masked.mask]

pvals_adjusted = multipletests(pvals_valid, method='fdr_bh')[1]

# put the result back in the right spots in the matrix
out = np.ones_like(pvals)
out[~pvals_masked.mask] = pvals_adjusted
out[pvals_masked.mask] = np.nan

out = -np.log10(out)

np.savetxt(outfile,out,fmt="%.3f",delimiter="\t")

