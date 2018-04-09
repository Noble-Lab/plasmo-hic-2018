import sys
import numpy as np
from hicutils import *
import logging
import gzip

def main():
    logging.basicConfig(level=logging.DEBUG)
    matfile_A = sys.argv[1] # normalized counts
    matfile_B = sys.argv[2] # normalized counts
    fithicfile_A = sys.argv[3] # normalized counts
    fithicfile_B = sys.argv[4] # normalized counts
    outfile = sys.argv[5]
    res = 10000

    all_chrs = read_chrs("/net/noble/vol2/home/katecook/proj/2015sexualStages/results/katecook/sample_info/chrs.lst")
    chrom_sizes = read_chrom_sizes("/net/noble/vol2/home/katecook/proj/2015sexualStages/results/katecook/sample_info/chr_sizes.tab")
    chrom_starts = chrom_sizes_to_starts(chrom_sizes,all_chrs)

    # load data
    matA = load_sparse(matfile_A,chrom_sizes,chrom_starts,res)
    matB = load_sparse(matfile_B,chrom_sizes,chrom_starts,res)
    
    fithic_expected_A = load_sparse(fithicfile_A,chrom_sizes,chrom_starts,res,col=9)
    fithic_expected_B = load_sparse(fithicfile_B,chrom_sizes,chrom_starts,res,col=9)
    
    observed_over_expected_A = (matA+1)/(fithic_expected_A+1)
    observed_over_expected_B = (matB+1)/(fithic_expected_B+1)
    
    ratios = (observed_over_expected_A+1)/(observed_over_expected_B+1)
    np.savetxt(outfile,ratios,delimiter='\t',fmt='%4f') 



if __name__ == "__main__":
    main()
