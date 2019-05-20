import numpy as np
import nibabel as nib


def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays]) 
    return data


def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)


def array_correlation(x, y, axis=0):

    """Column- or row-wise Pearson correlation between two arrays
    Computes sample Pearson correlation between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing correlations between two corresponding arrays with many
    variables (e.g., many voxels).
    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables
    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)
    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)
    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables
    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    # Center (de-mean) input variables
    x_demean = x - np.mean(x, axis=0)
    y_demean = y - np.mean(y, axis=0)

    # Compute summed product of centered variables
    numerator = np.sum(x_demean * y_demean, axis=0)

    # Compute sum squared error
    denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
                          np.sum(y_demean ** 2, axis=0))

    return numerator / denominator