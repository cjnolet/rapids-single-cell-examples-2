import cupy as cp
import numpy as np
from anndata import AnnData
import scipy.sparse
import cudf

n_rows = 350_000
n_cols = 100_000
mean = 2000
std_dev = 1000 # clipped to x > 0

density = 0.074727


def generate_cellxgene_naive(n_rows=70_000,
                             n_cols=20_000,
                             target_density=0.074727, # 7% density seems standard
                             row_degree_mean=2000, 
                             row_degree_std=1000,
                             include_genes=["ACE2", "TMPRSS2", "EPCAM"]):

    nnz = int(n_rows * n_cols * density)

    # Generate random degree matrix following a normal distribution
    degree = cp.abs(cp.random.normal(row_degree_mean, row_degree_std, size=n_rows).astype("int32"))+1

    # Weight the degree matrix so it sums to nnz
    weight = nnz / cp.sum(degree)
    degree = (degree * weight).astype("int32")

    # Account for floating point truncations
    new_nnz = cp.sum(degree).item()

    print("actual NNZ: %d" % new_nnz)
    print("actual density: %f" % (cp.sum(degree) / (n_rows * n_cols)))

    # Generate sorted CSR row pointer array
    rowptr = cp.zeros((n_rows+1,), dtype=cp.int32)
    cumsum = cp.cumsum(degree)
    rowptr[1:] = cumsum
    
    cols = cp.zeros((new_nnz,), dtype=cp.int32)
    for idx, start_idx in enumerate(rowptr[:n_rows]):
        stop_idx = rowptr[idx+1]
        d = stop_idx - start_idx

        if idx % 10_000 == 0 and idx != 0:
            print("Generated %d out of %d rows." % (idx, n_rows))
        rint = cp.random.randint(0, (n_cols-1), size=int(d))
        new_cols = cp.sort(rint)
        cols[start_idx:stop_idx] = new_cols
        
    vals = cp.random.negative_binomial(1, 
                                   .1, 
                                   size=new_nnz,
                                   dtype=cp.float32)

    var = cudf.Series(data=cp.arange(n_cols)).astype(str).to_pandas()
    obs = cudf.Series(cp.arange(n_rows)).astype(str).to_pandas()
    
    sp = scipy.sparse.csr_matrix((vals.get(), cols.get(), rowptr.get()), shape=(n_rows, n_cols))

    adata = AnnData(X=sp, var=var, obs=obs)

    
    var_names = cudf.Series(adata.var_names).astype(str).to_pandas()
    for idx, gene in enumerate(include_genes):
        var_names[idx] = gene

    adata.var_names = var_names
    
    return adata


