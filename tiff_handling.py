# These functions were originally coded by Allard Hendriksen (https://github.com/ahendriksen)
# Small modifications have been made to add functionality

from pathlib import Path
import tifffile
from tqdm import tqdm
import numpy as np
import torch


def load_stack(path, *, prefix="", skip=1, squeeze=False):
    """Load a stack of tiff files.

    Make sure that the tiff files are sorted *alphabetically*,
    otherwise it is not going to look pretty..

    :param path: path to directory containing tiff files
    :param skip: read every `skip' image
    :param squeeze: whether to remove any empty dimensions from image
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    path = Path(path).expanduser().resolve()

    # Only read every `skip' image:
    img_paths = sorted(path.glob(prefix+"*.tif"))[::skip]
    # Make a list containing
    if squeeze:
        imgs = [tifffile.imread(str(p)).squeeze() for p in tqdm(img_paths)]
    else:
        imgs = [tifffile.imread(str(p)) for p in tqdm(img_paths)]

    return np.array(imgs)
    

def save_stack(path, data, *, prefix="output", exist_ok=False, parents=False):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    for i, d in tqdm(enumerate(data), mininterval=1.0):
        output_path = path / f"{prefix}_{i:05d}.tif"
        tifffile.imsave(str(output_path), d)
