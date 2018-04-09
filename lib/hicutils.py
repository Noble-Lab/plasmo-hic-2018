import sys
import csv
import numpy as np
import scipy.sparse
import logging
import gzip
import math

def chrom_sizes_to_starts(chrom_sizes,all_chrs):
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

def index_to_chrloc(index,chrom_starts,res,all_chrs):
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

def read_chrom_sizes(infile):
    chrom_sizes = {}
    with open(infile) as fh:
        reader = csv.reader(fh,delimiter='\t')
        reader.next()
        for line in reader:
            #logging.debug(line)
            chrom_sizes[line[0]] = int(line[1])
    return chrom_sizes

def read_chrs(chrfile):
    fh = open(chrfile,'r')
    chrs = []
    for line in fh:
        chrs.append(line.rstrip())
    return chrs

def load_sparse(infile,chrom_sizes,chrom_starts,res,col=5):
    logging.info("loading matrix from %s" % infile)
    logging.info(chrom_starts)
    logging.info(chrom_sizes)
    logging.info(res)
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

def write_nelle_sparse(data,outfile,chrom_starts,all_chrs,res,floatdata=False):
    ofh = open(outfile,'w')
    n = data.shape[0]
    for i in range(n):
        for j in range(i,n):
            count = data[i,j]
            if count != 0:
                if floatdata:
                    ofh.write("%d\t%d\t%f\n" % (i,j,count))
                else:
                    ofh.write("%d\t%d\t%d\n" % (i,j,count))
    ofh.close()

def write_ferhat_sparse(data,outfile,chrom_starts,all_chrs,res,floatdata=False):
    ofh = open(outfile,'w')
    n = data.shape[0]
    for i in range(n):
        for j in range(i,n):
            count = data[i,j]
            if count != 0:
                chr_i,loc_i = index_to_chrloc(i,chrom_starts,res,all_chrs)
                chr_j,loc_j = index_to_chrloc(j,chrom_starts,res,all_chrs)
                loc_i = loc_i + res/2
                loc_j = loc_j + res/2
                if floatdata:
                    ofh.write("%s\t%d\t%s\t%d\t%f\n" % (chr_i,loc_i,chr_j,loc_j,count))
                else:
                    ofh.write("%s\t%d\t%s\t%d\t%d\n" % (chr_i,loc_i,chr_j,loc_j,count))
    ofh.close()

def extract_chromosome(data,chrom,chrom_sizes,chrom_starts,res):
    start = chrloc_to_index(chrom,0,chrom_starts,res)
    end = start + chrom_sizes[chrom]
    if scipy.sparse.isspmatrix_coo(data):
        data = scipy.sparse.csc_matrix(data)
    logging.debug("start: %d, end: %d" % (start,end))
    return data[start:end,start:end]

def put_back_chromosome(submat,mainmat,chrom,chrom_sizes,chrom_starts,res):
    start = chrloc_to_index(chrom,0,chrom_starts,res)
    end = start + chrom_sizes[chrom]
    logging.debug("start: %d, end: %d" % (start,end))
    assert submat.shape[0] == chrom_sizes[chrom]
    assert submat.shape[1] == chrom_sizes[chrom]
    mainmat[start:end,start:end] = submat
    return mainmat

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

# sets all bins between start[i] and end[i] to val
# ie filters out horizontal and vertical stripes
def filter_bins_1d(mat,starts,ends,val=np.nan):
    for start,end in zip(starts,ends):
        mat[start:end,:] = val
        mat[:,start:end] = val
    return mat

# sets the intersection of regions between start[i] and end[i]
# and start[j] and end[j] to val
# e.g. filters out all VSRM-VSRM interactions
def filter_bins_2d(mat,starts,ends,val=np.nan):
    for start_1,end_1 in zip(starts,ends):
        for start_2,end_2 in zip(starts,ends):
            #logging.debug("filtering 2d: %d %d %d %d" % (start_1,end_1,start_2,end_2) )
            mat[start_1:end_1,start_2:end_2] = val
            mat[start_2:end_2,start_1:end_1] = val
    return mat

def load_VSRM_locs(VSRMfile):
    fh = open(VSRMfile,'r')
    reader = csv.reader(fh,delimiter="\t")
    reader.next() # skip header
    pos = []
    for line in reader:
        chrom = "chr"+line[0]
        start = int(line[2])
        end = int(line[3])
        pos.append((chrom,start,end))
    return pos

def load_telomeric_VSRM_locs(VSRMfile):
    fh = open(VSRMfile,'r')
    reader = csv.reader(fh,delimiter="\t")
    reader.next() # skip header
    pos = []
    for line in reader:
        if line[7] == "telomeric":
            chrom = "chr"+line[0]
            start = int(line[2])
            end = int(line[3])
            pos.append((chrom,start,end))
    return pos

def symmetrize(mat):
    # assume matrix is upper triangular
    mat[np.tril_indices(mat.shape[0],-1)] = 0
    transposed = np.copy(mat.T)
    np.fill_diagonal(transposed,0)
    return mat + transposed

