from pathlib import Path
import tifffile
from tqdm import tqdm
import numpy as np
import torch


def load_stack(path, *, prefix="", skip=1, squeeze=False, dtype=None, stack_axis=0):
    """Load a stack of tiff files.

    Make sure that the tiff files are sorted *alphabetically*,
    otherwise it is not going to look pretty..

    :param path: path to directory containing tiff files
    :param skip: read every `skip' image
    :param squeeze: whether to remove any empty dimensions from image
    :param dtype: sets the type of the resulting array. All images will be cast to this type
    :param stack_axis: determines the dimension in the result where the image in the stack will be indexed
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    path = Path(path).expanduser().resolve()

    # Only read every `skip' image:
    img_paths = sorted(path.glob(prefix+"*.tif"))[::skip]
    img0 = tifffile.imread(str(img_paths[0]))
    if dtype is None:
        dtype = img0.dtype
    
    result_shape = np.insert(np.array(img0.shape), stack_axis, len(img_paths))
    result = np.empty(result_shape, dtype=dtype)
    for i, p in enumerate(tqdm(img_paths)):
        read_image = tifffile.imread(str(p)).astype(dtype=dtype, copy=False)
        if stack_axis == 0:
            result[i, ...] = read_image
        elif stack_axis == 1:
            result[:, i, ...] = read_image
        else:
            result[:, :, i] = read_image
    
    if squeeze:
        result = result.squeeze()
    
    return result
    

def save_stack(path, data, *, prefix="output", exist_ok=False, parents=False, stack_axis=0):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    for i in tqdm(range(data.shape[stack_axis])):
        output_path = path / f"{prefix}_{i:05d}.tif"
        tifffile.imsave(str(output_path), data.take(indices=i, axis=stack_axis))
