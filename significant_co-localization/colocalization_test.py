import sys
import argparse
import numpy as np
import scipy
import logging
import csv
import gzip
import random
import math
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from scipy.stats import norm


res = 10000
#all_chrs = ['chr1', 'chr2']
all_chrs = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14']


def main():
    logging.basicConfig(level = logging.DEBUG)
    
    #parser = argparse.ArgumentParser(description='Test a set of genes for colocalization in Hi-C data')
    #parser.add_argument('Gene list', metavar='genes', type=argparse.FileType('r'), nargs='1',
    #                help='tab-delimited list of genes to test for colocalization')
    #parser.add_argument('--test', choices=['hyper','WN','Paulsen'],
    #                help="test to perform. One of ['hyper','WN','Paulsen']")
    #parser.add_argument('--infile', type=argparse.FileType('r'), nargs='1',
    #                help="Hi-C data file. One of either contact counts (for hyper or Paulsen) or fit-hi-c (for WN)")
    #parser.add_argument()
    #parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                const=sum, default=max,
    #                help='sum the integers (default: find the max)')

    
    genelistfile = sys.argv[1]
    countfile = sys.argv[2]
    fithicfile = sys.argv[3]
    chrsizefile = sys.argv[4]
    n_trials = int(sys.argv[5])
    outfile = sys.argv[6]

    # read in data
    logging.info("Reading input data")
    chrom_sizes = read_chrom_sizes(chrsizefile)
    chrom_starts = chrom_sizes_to_starts(chrom_sizes)

    chrom_sizes_list = [chrom_sizes[c] for c in all_chrs]
    chrom_starts_list = [chrom_starts[c] for c in all_chrs] 

    genes = read_genes_from_tabfile(genelistfile)
    genes = sort_genes(genes)
    genebins = get_bins_for_genes(genes, chrom_starts)
    logging.debug('genebins' + str(genebins))

    #counts = read_counts(countfile)
    fdrs = read_fdr(fithicfile,chrom_sizes,chrom_starts,res,col=7)
    
    # do hypergeometric
    #TODO
    logging.info("skipping hypergeometric")
    
    # do WN
    logging.info("do WN test")
    boolcounts = (fdrs<=0.05).astype(bool)
    # count total number of observed interactions and total possible
    total_observed,total_possible = calculate_observed_interactions(genebins,boolcounts)
    #sys.exit()
    logging.info("total observed: %d total possible: %d" % (total_observed,total_possible))
    observed_ratio = float(total_observed)/total_possible
    # do permutations
    B = 0
    for i in range(n_trials):
        if i%500 == 0:
            logging.info('.')
        randbins = {}
        for c in all_chrs:
            n_genes = len(genebins[c])
            cstart = chrom_starts[c]
            cend = cstart + chrom_sizes[c]-1
            randbins[c] = []
            for j in range(n_genes):
                randbins[c].append(random.randint(cstart,cend))
        permuted_observed,permuted_possible = calculate_observed_interactions(randbins,boolcounts)
        permuted_ratio = float(permuted_observed)/permuted_possible
        if permuted_ratio >= observed_ratio:
            B = B + 1
    wn_pval = float(B)/n_trials
    logging.info("WN P-value: " + str(wn_pval))
    
    outfh = open(outfile,'w')
    outfh.write("%.3f\n" % (wn_pval))
    outfh.close()
    sys.exit()
    
    # do Paulsen
    
    #counts = nan_out_interchrom_zeros(counts,chrom_starts_list,chrom_sizes_list)
    #counts = nan_out_all_zeros(counts)
    zscored_counts = zscore_counts_by_distance(counts,chrom_starts_list,chrom_sizes_list)

    outtest = open("arg.txt",'w')
    outtest.write("type\tcount\n")
    
    observed_counts = get_counts(genebins, zscored_counts)
    #logging.debug('counts_obs:' + str(observed_counts))
    t_obs = calculate_t_statistic(observed_counts)
    logging.debug('t_obs: ' + str(t_obs))
    
    for c in observed_counts:
        outtest.write("obs\t%.3f\n" % c)
     
    permuted_t = []
    for i in range(n_trials):
        permuted_genes = select_permuted_set(genebins, chrom_sizes)
        #logging.debug("permuted bins: " + str(permuted_genes))
        permuted_counts = get_counts(permuted_genes, zscored_counts)
        #logging.debug('permuted counts:' + str(permuted_counts))
        if i < 100:
            for c in permuted_counts:
                outtest.write("perm\t%.3f\n" % c)
        t = calculate_t_statistic(permuted_counts)
        #logging.debug("permuted t: " + str(t))
        permuted_t.append(t)
    
    #outtest.close()
    
    paulsen_pval = calculate_pval(t_obs, permuted_t)
    logging.info("Paulsen P-value: " + str(paulsen_pval))
    
    outfh = open(outfile,'w')
    outfh.write("%.3f\t%.3f\n" % (wn_pval,paulsen_pval))
    outfh.close()

def calculate_observed_interactions(genebins,boolcounts):
    total_observed = 0
    total_possible = 0
    for target_chrom in all_chrs:
        for target_bin in genebins[target_chrom]:
            other_bins = []
            for other_chrom in all_chrs:
                if other_chrom == target_chrom:
                    continue
                other_bins += genebins[other_chrom]
            logging.debug("target_bin: " + str(target_bin))
            logging.debug("other_bins: " + str(other_bins))
            logging.debug("observed: " + str(boolcounts[target_bin,other_bins]))
            total_observed += np.sum(boolcounts[target_bin,other_bins])
            total_possible += len(boolcounts[target_bin,other_bins])
    return total_observed,total_possible

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return list(rows), list(cols)

#note: only does upper triangular
def interchr_indices_from_chr(shape,cstart,cend):
    boolmat = np.zeros(shape)
    boolmat[cstart:cend,cend:] = True
    rows,cols = np.where(boolmat) 
    return list(rows),list(cols)

def symmetrize(mat):
    assert is_upper_tri(mat)
    mat[np.tril_indices(mat.shape[0],-1)] = 0
    transposed = np.copy(mat.T)
    np.fill_diagonal(transposed,0)
    return mat + transposed

def nan_out_all_zeros(count_matrix):
    out_matrix = count_matrix.copy()
    out_matrix[np.where(out_matrix<1)] = np.nan
    return out_matrix

def nan_out_interchrom_zeros(count_matrix,chr_starts,chr_sizes):
    out_matrix = np.empty_like(count_matrix)
    out_matrix[:] = np.nan
    interchrom_rows = []
    interchrom_cols = []
    for start,size in zip(chr_starts,chr_sizes):
        end = start + size
        r,c = interchr_indices_from_chr(count_matrix.shape,start,end)
        interchrom_rows = interchrom_rows + r
        interchrom_cols = interchrom_cols + c
        interchrom_cols = interchrom_cols + r
        interchrom_rows = interchrom_rows + c
        out_matrix[start:end,start:end] = count_matrix[start:end,start:end]
        logging.debug('out_matrix:\n' + str(out_matrix))
    logging.debug("interchrom_rows")
    logging.debug(interchrom_rows)
    logging.debug("interchrom_cols")
    logging.debug(interchrom_cols)
    interchr_counts = count_matrix[interchrom_rows,interchrom_cols]
    interchr_counts[np.where(interchr_counts<1)] = np.nan
    out_matrix[interchrom_rows,interchrom_cols] = interchr_counts
    return out_matrix
    

def zscore_counts_by_distance(count_matrix,chr_starts,chr_sizes):
    zscored_matrix = np.empty_like(count_matrix)
    zscored_matrix[:] = np.nan
    # do interchromosomal
    interchrom_rows = []
    interchrom_cols = []
    for start,size in zip(chr_starts,chr_sizes):
        end = start + size
        r,c = interchr_indices_from_chr(count_matrix.shape,start,end)
        interchrom_rows = interchrom_rows + r
        interchrom_cols = interchrom_cols + c
    interchr_counts = count_matrix[interchrom_rows,interchrom_cols]
    zscored_interchr_counts = zscore_counts(interchr_counts)
    zscored_matrix[interchrom_rows,interchrom_cols] = zscored_interchr_counts
    # do each distance
    for i in range(count_matrix.shape[0]):
        rows = []
        cols = []
        for start,size in zip(chr_starts,chr_sizes):
            end = start + size
            chr_count_matrix = count_matrix[start:end,start:end]
            chr_rows,chr_cols = kth_diag_indices(chr_count_matrix,-i)
            rows.extend([x + start for x in chr_rows])
            cols.extend([x + start for x in chr_cols])
        length_counts = list(count_matrix[rows,cols])
        zscored_counts = zscore_counts(length_counts)
        zscored_matrix[rows,cols] = zscored_counts
    zscored_matrix[np.tril_indices(zscored_matrix.shape[0],-1)] = 0
    zscored_matrix = symmetrize(zscored_matrix)
    #np.savetxt("zscored_mat.txt",symmetrize(zscored_matrix)) 
    return zscored_matrix


# do this per distance
def zscore_counts(interaction_counts):
    if len(interaction_counts) < 2:
        return [0]
    m = np.nanmean(interaction_counts)
    sd = np.nanstd(interaction_counts)
    zscores = (interaction_counts - m)/sd
    return list(zscores)


def calculate_t_statistic(interaction_counts):
    M = len(interaction_counts)
    #logging.debug(M)
    #logging.debug(np.sum(interaction_counts))
    t = np.nansum(interaction_counts)/M
    return t

def calculate_intergene_distances(genes):
    for chrom in chrs:
        chrgenes = genes[chrom]
        # sort by start position
        for gene in chrgenes:
            pass 

# sort by chromosome and location
def sort_genes(genes):
    sortedgenes = {}
    for c in all_chrs:
        sortedgenes[c] = []
    for c in genes:
        for gene in genes[c]:
            sortedgenes[c].append(gene)
    for c in all_chrs:
        sortedgenes[c] = sorted(sortedgenes[c])
    return sortedgenes

def get_distances(bins):
    # assume input bins are all on the same chromosome
    distances = []
    for i in range(len(bins)-1):
        d = bins[i+1] - bins[i]
        distances.append(d)
    return distances

def select_permuted_set(bins,chrom_sizes):
    permuted_bins = {}
    #logging.debug('bins: ' + str(bins))
    for c in all_chrs:
        cbins = bins[c]
        distances = get_distances(cbins)
        total_distance = sum(distances)
        #logging.debug("distances: " + str(distances))
        random.shuffle(distances)
        #logging.debug("randomized distances: " + str(distances))
        cur_bin = pick_start(total_distance,chrom_sizes[c])
        #logging.debug("start: " + str(cur_bin))
        if len(cbins) == 0:
            pbins = []
        else:
            pbins = [cur_bin]
            for d in distances:
                new_bin = cur_bin + d
                pbins.append(new_bin)
                cur_bin = new_bin
        permuted_bins[c] = pbins
    return permuted_bins

def pick_start(total_distance,chr_size):
    length_to_pick_from = chr_size - total_distance
    start = random.randint(0,length_to_pick_from)
    return start
    

def calculate_pval(t_obs, permuted_t):
    summed = 0.
    R = len(permuted_t)
    for t in permuted_t:
        if t >= t_obs:
            summed = summed + 1
    p = (summed + 1)/(R + 1)
    return p

def get_bins_for_genes(genes,chrom_starts):
    bins = {}
    for c in all_chrs:
        cgenes = genes[c]
        cbins = []
        for (start,end) in cgenes:
            i = chrloc_to_index(c,start,chrom_starts,res)
            j = chrloc_to_index(c,end,chrom_starts,res)
            #logging.debug(str(i) + " " + str(j))
            if j>i:
                for k in range(i,j+1):
                    if k not in cbins:
                        cbins.append(k)
            else:
                if i not in cbins:
                    cbins.append(i)
        bins[c] = sorted(cbins)
    return bins

def get_counts(bins, counts):
    out = []
    allbins = []
    for c in all_chrs:
        allbins = allbins + bins[c]
    for bin1 in allbins:
        for bin2 in allbins:
            if bin1==bin2:
                continue
            out.append(counts[bin1,bin2])
    #for c in all_chrs:
    #    cbins = bins[c]
    #    for bin1 in cbins:
    #        for bin2 in cbins:
    #            if bin1 == bin2:
    #                continue
    #            out.append(counts[bin1,bin2])
    return out

def chrom_sizes_to_starts(chrom_sizes):
    start_index = 0
    end_index = 0
    chrom_starts = {}
    chrom_ends = {}
    for chrom in all_chrs:
        end_index = start_index + chrom_sizes[chrom]
        #logging.debug([str(x) for x in [chrom,start_index,end_index]])
        chrom_starts[chrom] = start_index
        chrom_ends[chrom] = end_index
        start_index = end_index
    return chrom_starts

def index_to_chrloc(index,chrom_starts,res):
    chrom = 0
    loc = 0
    for i,c in enumerate(all_chrs):
        #logging.debug([str(x) for x in [index,c,chrom_starts[c]]])
        if index >= chrom_starts[c]:
            chrom = c
            loc = index - chrom_starts[c]
            loc = loc * res
    return (chrom,loc)

def chrloc_to_index(chrom,loc,chrom_starts,res):
    firstbin = math.floor(loc / res)
    index = int(chrom_starts[chrom] + firstbin)
    return index

####################################################
# Read input
####################################################

def read_genes_from_tabfile(infile):
    genes = {}
    for c in all_chrs:
        genes[c] = []
    with open(infile) as fh:
        reader = csv.reader(fh,delimiter='\t')
        for line in reader:
            #logging.debug(line)
            chrom = 'chr' + line[1]
            start = int(line[2])
            end = int(line[3])
            genes[chrom].append((start,end))
    return genes

def read_genes_from_bedfile(infile):
    genes = {}
    for c in all_chrs:
        genes[c] = []
    with open(infile) as fh:
        reader = csv.reader(fh,delimiter='\t')
        for line in reader:
            chrom = line[0]
            start = int(line[1])
            end = int(line[2])
            genes[chrom].append((start,end))
    return genes

def read_chrom_sizes(infile):
    chrom_sizes = {}
    with open(infile) as fh:
        reader = csv.reader(fh,delimiter='\t')
        reader.next()
        for line in reader:
            #logging.debug(line)
            chrom_sizes[line[0]] = int(line[1])
    return chrom_sizes

def read_counts(infile):
    counts = np.loadtxt(infile)
    return counts

def read_fdr(infile,chrom_sizes,chrom_starts,res,col=5):
    logging.info("loading matrix from %s" % infile)
    logging.debug(chrom_starts)
    logging.debug(chrom_sizes)
    logging.debug(res)
    if infile.endswith('gz'):
        fh = gzip.open(infile)
    else:
        fh = open(infile,'r')
    reader = csv.reader(fh,delimiter='\t')
    x = []
    y = []
    counts = []
    i = 0
    for line in reader:
        if i % 10000 == 0 : #FIXME: add check for logging level
            sys.stderr.write(".")
        i = i + 1
        chr1,mid1,chr2,mid2 = line[0:4]
        mid1 = int(mid1)
        mid2 = int(mid2)
        count = line[col-1]
        count = float(count)
        index_1 = chrloc_to_index(chr1,mid1,chrom_starts,res)
        index_2 = chrloc_to_index(chr2,mid2,chrom_starts,res)
        if index_2 < index_1:
            save = index_2
            index_2 = index_1
            index_1 = save
        assert index_1 <= index_2, "not upper triangular!!! %s %d %s %d (%d,%d)" % (chr1,mid1,chr2,mid2,index_1,index_2)
        x.append(index_1)
        y.append(index_2)
        counts.append(count)
        #logging.debug(", ".join([str(t) for t in [chr1,mid1,index_1,chr2,mid2,index_2,count]]))
    logging.debug("done")
    logging.debug("creating arrays")
    last_bin = sum(chrom_sizes.values()) - 1
    x.append(last_bin)
    y.append(last_bin)
    counts.append(0)
    x_ary = np.array(x)
    y_ary = np.array(y)
    counts_ary = np.array(counts)
    logging.debug("array min/max: %e %e" % (np.min(counts_ary),np.max(counts_ary)))
    logging.debug("Creating matrix")
    mat = scipy.sparse.coo_matrix((counts_ary, (x_ary, y_ary)))
    logging.debug("matrix min/max: %e %e" % (mat.min(), mat.max()))
    return np.array(mat.todense())


def is_upper_tri(X):
    diag = X.diagonal().sum()
    if np.tril(X, -1).sum():
        return False
    else:
        return True


####################################################
# Tests
####################################################

def test_calculate_pval():
    t_obs_test = 4.
    permuted_t_test = [1.,2.,3.,4.,5.,6.,7.]
    p_val_test = calculate_pval(t_obs_test,permuted_t_test)
    assert_almost_equal(p_val_test,0.625)

def test_chrloc_to_index():
    chrom_starts = chrom_sizes_to_starts( read_chrom_sizes('/net/noble/vol2/home/katecook/proj/2015sexualStages/results/katecook/sample_info/chr_sizes.tab') )
    chrom = 'chr1'
    loc = 10
    index = chrloc_to_index(chrom,loc,chrom_starts,1)
    print(index)
    assert index == 10
    chrom = 'chr2'
    loc = 10
    index = chrloc_to_index(chrom,loc,chrom_starts,1)
    print(index)
    assert index == 75, index


def test_zscore_counts():
    test_data = np.array([0.,1.,2.,3.,4.,5.])
    zscores = zscore_counts(test_data)
    #logging.debug(zscores)
    assert_array_almost_equal(zscores, np.array([-1.46385011, -0.87831007, -0.29277002, 0.29277002, 0.87831007, 1.46385011]) )

def test_index_to_chrloc():
    chrom_starts = chrom_sizes_to_starts( read_chrom_sizes('/net/noble/vol2/home/katecook/proj/2015sexualStages/results/katecook/sample_info/chr_sizes.tab') )
    index = 24
    chrom,loc = index_to_chrloc(index,chrom_starts,1)
    assert chrom == 'chr1'
    assert loc == 24
    index = 75
    chrom,loc = index_to_chrloc(index,chrom_starts,1)
    assert chrom == 'chr2'
    assert loc == 10, loc

def test_sort_genes():
    test_genes = {}
    sorted_genes = {}
    for c in all_chrs:
        test_genes[c] = []
        sorted_genes[c] = []
    test_genes['chr1'] = [(1,10), (20,30), (15,600)]
    test_genes['chr3'] = [(60,70), (1,2)]
    sorted_genes['chr1'] = [(1,10), (15,600), (20,30)]
    sorted_genes['chr3'] = [(1,2), (60,70)]
    output_genes = sort_genes(test_genes)
    #logging.debug(test_genes)
    #logging.debug(sorted_genes) 
    assert output_genes == sorted_genes

def test_t_stat():
    test_counts = np.array([0.,1.,2.,3.,4.,5.])
    t = calculate_t_statistic(test_counts)
    #logging.debug(t)
    assert t == 2.5

def test_interchr_indices_from_chr():
    shape = (8,8)
    start = 0
    end = 3
    rows,cols = interchr_indices_from_chr(shape,start,end)
    #logging.debug(rows)
    #logging.debug(cols)
    assert rows == [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    assert cols == [3,4,5,6,7,3,4,5,6,7,3,4,5,6,7]
    start = 3
    end = 5
    rows,cols = interchr_indices_from_chr(shape,start,end)
    #logging.debug(rows)
    #logging.debug(cols)
    assert rows == [3,3,3,4,4,4]
    assert cols == [5,6,7,5,6,7]
    start = 5
    end = 8
    rows,cols = interchr_indices_from_chr(shape,start,end)
    logging.debug(rows)
    logging.debug(cols)
    assert rows == []
    assert cols == []

def test_zscore_counts_by_distance_singlechr():
    input_matrix = np.array([[ 1,  5,  8, 10],
       [ 5,  2,  6,  9],
       [ 8,  6,  3,  7],
       [10,  9,  7,  4]],dtype=np.float64)
    result = zscore_counts_by_distance(input_matrix,[0],[5])
    zscored_matrix = np.array([[-1.34164079, -1.22474487, -1.        ,  0.        ],
       [-1.22474487, -0.4472136 ,  0.        ,  1.        ],
       [-1.        ,  0.        ,  0.4472136 ,  1.22474487],
       [ 0.        ,  1.        ,  1.22474487,  1.34164079]])
    assert_array_almost_equal(zscored_matrix,result)

def test_is_upper_tri():
    # test upper tri
    mat = np.array([[1,4,5],[0,2,6],[0,0,3]])
    assert(is_upper_tri(mat))
    # test symm
    mat = symmetrize(mat)
    assert(not is_upper_tri(mat))
    # test lower tri
    mat = np.array([[3,0,0],[1,2,0],[1,4,5]])
    assert(not is_upper_tri(mat))
    

def test_symmetrize():
    mat = np.array([[1,4,5],[0,2,6],[0,0,3]])
    logging.debug('\n'+str(mat))
    result = symmetrize(mat)
    logging.debug('\n'+str(result))
    result2 = symmetrize2(mat)
    logging.debug('\n'+str(result2))
    symmat = np.array([[1,4,5],[4,2,6],[5,6,3]])
    logging.debug('\n'+str(symmat))
    assert_array_almost_equal(result,symmat)
    assert_array_almost_equal(result2,symmat)
    assert_array_almost_equal(result,result2)

def test_symmetrize_zerodiag():
    mat = np.array([[0,4,5],[0,0,6],[0,0,0]])
    logging.debug('\n'+str(mat))
    result = symmetrize(mat)
    logging.debug('\n'+str(result))
    result2 = symmetrize2(mat)
    logging.debug('\n'+str(result2))
    symmat = np.array([[0,4,5],[4,0,6],[5,6,0]])
    logging.debug('\n'+str(symmat))
    assert_array_almost_equal(result,symmat)
    assert_array_almost_equal(result2,symmat)
    assert_array_almost_equal(result,result2)


def test_zscore_counts_by_distance_multichr():
    input_matrix = np.array([[1,1,1,2,3],[1,2,7,8,9],[1,7,3,2,1],[2,8,2,4,3],[3,9,1,3,5]], dtype=np.float64)
    chr_starts = [0,2]
    chr_sizes = [2,3]
    result = zscore_counts_by_distance(input_matrix,chr_starts,chr_sizes)
    zscored_matrix = np.array([[-1.41421356, -1.22474487, -1.28653504, -0.96490128, -0.64326752],
       [-1.22474487, -0.70710678,  0.64326752,  0.96490128,  1.28653504],
       [-1.28653504,  0.64326752,  0.        ,  0.        ,  0.        ],
       [-0.96490128,  0.96490128,  0.        ,  0.70710678,  1.22474487],
       [-0.64326752,  1.28653504,  0.        ,  1.22474487,  1.41421356]])
    logging.debug('\n' + str(result))
    logging.debug('\n' + str(zscored_matrix))
    assert_array_almost_equal(zscored_matrix,result)

def test_nan_out_interchrom_zeros():
    input_matrix = np.array([[1,0,1,2,0],[0,2,7,8,0],[1,7,3,2,1],[2,8,2,4,3],[0,0,1,3,5]], dtype=np.float64)
    logging.debug('input\n' + str(input_matrix))
    chr_starts = [0,2]
    chr_sizes = [2,3]
    nand_out = np.array([[1,0,1,2,np.nan],[0,2,7,8,np.nan],[1,7,3,2,1],[2,8,2,4,3],[np.nan,np.nan,1,3,5]], dtype=np.float64)
    result = nan_out_interchrom_zeros(input_matrix,chr_starts,chr_sizes)
    logging.debug('\n' + str(result))
    logging.debug('\n' + str(nand_out))
    assert_array_almost_equal(nand_out,result)


### simulations

def run_clustered():
    n_trials,n_genes,n_permutations,chrom_starts,chrom_sizes,zscored_counts,outfh = initialize_trials()

    for trial in range(n_trials):
        logging.info("trial %d" % trial)
        genebins = get_normal_bins(n_genes,chrom_starts,chrom_sizes) 
        observed_counts = get_counts(genebins, zscored_counts)
        t_obs = calculate_t_statistic(observed_counts)
        # do permutations
        permuted_t = []
        for p in range(n_permutations):
            permuted_genes = select_permuted_set(genebins, chrom_sizes)
            logging.debug("permuted bins: " + str(permuted_genes))
            permuted_counts = get_counts(permuted_genes, zscored_counts)
            t = calculate_t_statistic(permuted_counts)
            permuted_t.append(t)

        pval = calculate_pval(t_obs, permuted_t)
        outfh.write(str(pval) + '\n')
    outfh.close()

def initialize_trials():
    logging.basicConfig(level = logging.INFO)
    
    countfile = sys.argv[1]
    chrsizefile = sys.argv[2]
    outfile = sys.argv[3]

    n_trials = 1000
    n_permutations = 1000
    n_genes = 5 # per chromosome

    # read in data    
    chrom_sizes = read_chrom_sizes(chrsizefile)
    chrom_starts = chrom_sizes_to_starts(chrom_sizes)
    chrom_sizes_list = [chrom_sizes[c] for c in all_chrs]
    chrom_starts_list = [chrom_starts[c] for c in all_chrs]
    endbin = chrom_starts[all_chrs[-1]] + chrom_sizes[all_chrs[-1]]

    # transform counts
    counts = read_counts(countfile)
    zscored_counts = zscore_counts_by_distance(counts,chrom_starts_list,chrom_sizes_list)

    # open outfile
    outfh = open(outfile, 'w')
    return n_trials,n_genes,n_permutations,chrom_starts,chrom_sizes,zscored_counts,outfh

def run_uniform():
    n_trials,n_genes,n_permutations,chrom_starts,chrom_sizes,zscored_counts,outfh = initialize_trials()
    
    
    for trial in range(n_trials):
        logging.info("trial %d" % trial)
        # select random set of genes
        genebins = get_uniform_bins(n_genes,chrom_starts,chrom_sizes) 
        observed_counts = get_counts(genebins, zscored_counts)
        t_obs = calculate_t_statistic(observed_counts)
        # do permutations
        permuted_t = []
        for p in range(n_permutations):
            permuted_genes = select_permuted_set(genebins, chrom_sizes)
            logging.debug("permuted bins: " + str(permuted_genes))
            permuted_counts = get_counts(permuted_genes, zscored_counts)
            t = calculate_t_statistic(permuted_counts)
            permuted_t.append(t)

        pval = calculate_pval(t_obs, permuted_t)
        outfh.write(str(pval) + '\n')
    outfh.close()

def get_uniform_bins(n_genes,chrom_starts,chrom_sizes):
    bins = {}
    for c in all_chrs:
        start = chrom_starts[c]
        end = start + chrom_sizes[c]
        if chrom_sizes[c] > 0:
            logging.debug("%d %d" % (start,end))
            cbins = random.sample(xrange(start,end),n_genes)
        else:
            cbins = []
        bins[c] = sorted(cbins)
    return bins

def get_normal_bins(n_genes,chrom_starts,chrom_sizes):
    bins = {}
    for c in all_chrs:
        start = chrom_starts[c]
        end = start + chrom_sizes[c]
        if chrom_sizes[c] > 0:
            # select center
            center = random.choice(range(start+5,end-5))
            logging.debug("center: %d" % center)
            # generate normal probabilities
            sd = chrom_sizes[c] / 5 
            probabilities = norm.pdf(range(start,end),loc=center,scale=sd)
            probabilities = probabilities / sum(probabilities)
            # select genes
            cbins = np.random.choice(range(start,end),size=n_genes,replace=False,p=probabilities)
        else:
            cbins = []
        bins[c] = sorted(cbins)
    return bins


###


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    #test_is_upper_tri()
    main()
