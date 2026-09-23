from __future__ import annotations

import itertools
from typing import cast, overload, Literal

import torch

@overload
def slice_sparse_tensor(tensor: torch.Tensor, dim0_bounds: tuple[int, int], trim_columns: Literal[True]) -> tuple[torch.Tensor, int]: ...

@overload
def slice_sparse_tensor(tensor: torch.Tensor, dim0_bounds: tuple[int, int], trim_columns: Literal[False]=False) -> torch.Tensor: ...

@torch.jit.script
def slice_sparse_tensor(tensor: torch.Tensor, dim0_bounds: tuple[int, int], trim_columns: bool=False) -> torch.Tensor | tuple[torch.Tensor, int]:
    """
    Slice the 1st dimension (rows) in a CSR sparse array.

    Useful for slicing the Lanczos matrix by TRs and finding which parts of the
    stimulus are important.
    """
    assert tensor.layout == torch.sparse_csr, "only works with CSR sparse layout"
    assert tensor.ndim == 2, f"only works with 2-D tensors but received {tensor.ndim}-D"
    assert (dim0_bounds[0] >= 0) and (dim0_bounds[1] <= tensor.shape[0]), f"slice indices out of bounds {dim0_bounds}"

    sliced_crow_indices = tensor.crow_indices()[dim0_bounds[0]:dim0_bounds[1]+1] # in the original matrix
    new_crow_indices = sliced_crow_indices - sliced_crow_indices.min() # in the new (sliced) matrix
    new_col_indices = tensor.col_indices()[sliced_crow_indices[0]:sliced_crow_indices[-1]]
    new_values = tensor.values()[sliced_crow_indices[0]:sliced_crow_indices[-1]]
    new_shape = (dim0_bounds[1] - dim0_bounds[0], tensor.shape[1])

    if trim_columns:
        # Trim unused (i.e. zero-valued) columns from the start and end
        if new_col_indices.shape[0] == 0:
            old_start_col = 0 # this would be a no-op if you add it back
            new_shape = (new_shape[0], 0) # dim. 0 still depends on the num. of rows
        else:
            old_start_col = int(new_col_indices.min().item()) # smallest column number in the original (`tensor`) that is in the slice
            new_col_indices = new_col_indices - old_start_col
            new_shape = (new_shape[0], int(new_col_indices.max().item())+1)
    else:
        # This branch should never be hit, but is required for the Pytorch JIT compiler
        old_start_col = 0

    new_tensor = torch.sparse_csr_tensor(new_crow_indices,
                                         col_indices=new_col_indices,
                                         values=new_values,
                                         size=new_shape,
                                         dtype=tensor.dtype,
                                         device=tensor.device)

    if trim_columns:
        return new_tensor, old_start_col

    return new_tensor

# check this with `pytest torch_sparse_utils.py`
def test_slice_sparse_tensor():
    shapes = [(1, 1), (1, 100), (100, 1), (1000, 3000)]

    # `None` = random length
    slice_lens = [0, 1, None, None, None, None, None, None, None, None]
    sparsities = [0.0, 0.1, 0.3, 0.5]

    for shape, slice_len, sparsity in itertools.product(shapes, slice_lens, sparsities):
        nnz = int(shape[0] * shape[1] * sparsity) # num. of nonzero entries
        nonzero_rows = torch.randint(shape[0], size=(nnz,))
        nonzero_cols = torch.randint(shape[1], size=(nnz,))
        nonzero_entries = torch.randn(size=(nnz,))
        orig_dense = torch.zeros(shape)
        orig_dense[nonzero_rows, nonzero_cols] = nonzero_entries
        orig_sparse = orig_dense.to_sparse_csr()

        # Test without trimming columns
        if slice_len is None:
            slice_len = int(torch.randint(shape[0], size=()))
        slice_start = int(torch.randint(max(shape[0]-slice_len, 1), size=()))
        slice_end = slice_start + slice_len
        sliced_sparse = slice_sparse_tensor(orig_sparse, dim0_bounds=(slice_start, slice_end), trim_columns=False)
        assert torch.allclose(orig_sparse.to_dense()[slice_start:slice_end],
                              sliced_sparse.to_dense())

        # Test with trimming columns
        sliced_sparse, old_start_col = slice_sparse_tensor(orig_sparse, dim0_bounds=(slice_start, slice_end), trim_columns=True)
        sliced_dense = torch.zeros_like(orig_sparse.to_dense())[slice_start:slice_end]
        sliced_dense[:, old_start_col:old_start_col+sliced_sparse.shape[1]] = sliced_sparse.to_dense()
        assert torch.allclose(orig_sparse.to_dense()[slice_start:slice_end],
                              sliced_dense)
