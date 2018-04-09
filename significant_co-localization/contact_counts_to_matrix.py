import sys
import contact_counts
import numpy as np
import logging


def symmetrize(a):
    a[np.tril_indices_from(a,-1)] = 0
    return a + a.T - np.diag(a.diagonal())


def main():
    logging.basicConfig(level=logging.DEBUG)
    binfile = sys.argv[1]
    precalculated_length_file = sys.argv[2]
    matfile = sys.argv[3] # normalized counts
    outfile = sys.argv[4]

    # set up bins   
    logging.info("setting up bins")
    (allBins,allBins_reversed,badBins) = contact_counts.generate_bins(binfile,0.5)

    # read input contact count matrices
    logging.info("creating matrix")
    matrix = contact_counts.contactCountMatrix(allBins,allBins_reversed,badBins)
    logging.info("loading matrix data")
    matrix.load_from_file(matfile)
    
    data = symmetrize(matrix.data.todense())
    #data[data==0] = np.nan

    # make matrix
    logging.info("saving output")
    np.savetxt(outfile,data,delimiter='\t')



if __name__ == "__main__":
    main()

