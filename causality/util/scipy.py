from itertools import izip
import numpy as np

# See https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py#L96.
def unique(mat, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique elements of a sparse matrix.

    Returns the sorted unique elements of a sparse matrix. There are two
    optional outputs in addition to the unique elements: the indices of the
    input matrix that give the unique values, and the indices of the unique
    matrix that reconstruct the input matrix.

    Parameters
    ----------
    mat : sparse matrix
        Input matrix. This will be converted to the CSR representation
        internally.
    return_index : bool, optional
        If True, also return the indices of `mat` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `mat`.
    return_counts : bool, optional
        If True, also return the number of times each unique value comes up
        in `mat`.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.

        Note that, because the matrix is sparse, the full array of indices is
        not returned. Instead, an array i is returned such that, given an empty
        sparse matrix m with the same number of columns as there were elements
        in mat, setting m[0, i[0]] = unique[i[1]] will reproduce the original
        matrix.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    numpy.lib.arraysetops.unique : Basis for this function, but only works for
                                   dense matrices/arrays.
    """
    # Convert to CSR because this format has a .data matrix, which is the main
    # thing we need, and which is stored in sorted order of rows -> columns.
    # This means that np.unique returns the indices and inverse in terms of a
    # sensibly linearized mat. (The nonzero indices are also returned in
    # row -> column order, which is useful for the return_inverse and
    # return_index options.) Also, CSR is fairly memory-efficient and quick to
    # convert to from other formats.
    mat = mat.tocsr()
    size = mat.shape[0] * mat.shape[1] # mat.size just gives nnz

    unique_data = np.unique(mat.data, return_index, return_inverse,
                            return_counts)

    # If there are no zeros, we can just pretend we're operating on a normal
    # dense array. All we have to do then is check whether we need to adapt the
    # inverse return value to our special sparse inverse format.
    if mat.nnz == size:
        if return_inverse:
            inv_index = (2 if return_index else 1)
            inverse = np.vstack((range(size), unique_data[inv_index]))
            unique_data = list(unique_data)
            unique_data[inv_index] = inverse
            unique_data = tuple(unique_data)
        return unique_data

    # OK, there are some zeros.
    # Our lives are simplest if the only thing requested was the unique values.
    if not isinstance(unique_data, tuple):
        # We got here because there are zeros, so we know 0 should be in the
        # list of unique values.
        return np.insert(unique_data, 0, 0.0)

    # If more values were requested, process other return values in the tuple
    # as necessary.
    unique_data = list(reversed(unique_data))
    unique_values = unique_data.pop()
    unique_values = np.insert(unique_values, 0, 0.0)
    ret = (unique_values,)

    # Offset returned indices to account for missing zero entries.
    if return_index or return_inverse:
        if return_index:
            indices = unique_data.pop()
        if return_inverse:
            inverse = unique_data.pop()

            # We're going to use inverse[0] as the array indices at which
            # values in the original matrix reside, and inverse[1] as the
            # indices in the unique array from which to draw those values.
            # We must add 1 to inverse[1] to account for the 0 in the initial
            # position.

            # The indices for the inverse matrix aren't accounting for the
            # presence of a zero value at the start of the list.
            inverse_unique_indices = inverse + 1
            # Initialize positions in original matrix to values' current
            # positions in the inverse array. As we detect 0 values in the
            # original matrix, we'll increase these indices accordingly.
            inverse_orig_pos_indices = np.array(range(len(inverse)))

        first_zero = None
        offset = 0
        mat.sort_indices()
        nonzero = mat.nonzero()

        for i, (row, col) in enumerate(izip(nonzero[0], nonzero[1])):
            offset_i = i + offset
            flattened_index = row * mat.shape[1] + col
            difference = flattened_index - offset_i
            if difference > 0: # We've found one or more zero entries!
                if return_index:
                    indices[np.where(indices >= offset_i)] += difference
                    if first_zero is None:
                        first_zero = i
                        indices = np.insert(indices, 0, first_zero)
                if return_inverse:
                    inverse_orig_pos_indices[
                        np.where(inverse_orig_pos_indices >= offset_i)
                        ] += difference
                offset += difference

        if return_index:
            ret += (indices,)

        if return_inverse:
            inverse = np.vstack((inverse_orig_pos_indices,
                                 inverse_unique_indices))
            ret += (inverse,)

    # Add counts for 0 value.
    if return_counts:
        counts = unique_data.pop()
        counts = np.insert(counts, 0, size - mat.nnz)
        ret += (counts,)

    return ret
