"""
This script calculates the 'Synthetic Galaxy Distance' between two datasets.

Released under the AGPLv3 licence.

@AUTHOR Mike Smith
"""

import matplotlib as mpl
mpl.use("Agg")

import argparse
import numpy as np
from scipy.stats import rankdata, ks_2samp, wasserstein_distance, gaussian_kde
from scipy.optimize import curve_fit
from glob import glob
import matplotlib.pyplot as plt
import mpl_scatter_density
import argparse
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import sigma_clipped_stats

def to_magnitude(flux):
    return 22.5 - 2.5 * np.log10(flux)

def rev_magnitude(mag):
    return 10**((22.5 - mag)/2.5)

def create_circular_mask(h, w, center=None, radius=None):
    """
    From https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def get_aperture_magnitude(x, radius=6, correct=True):
    """
    Approximate flux by taking the sum of pixel values in an image.
    """
    bck = np.median(np.stack((x[:20,:20],x[:20,-20:],x[-20:,:20],x[-20:,-20:])))
    mask = create_circular_mask(x.shape[0], x.shape[1], radius=radius)
    if correct:
        x = np.sum(x[mask]) - (bck*np.sum(mask))
    else:
        x = np.sum(x[mask])
    return x

def get_size(gal, arcsec_per_pixel=0.262):
    """
    Get half-light radius of gal, assuming that centre of image == centre of gal.

    Here we assume that the galaxies are in linear flux units.
    """
    gal = (gal - np.min(gal))/(np.max(gal) - np.min(gal))

    initial_rad = 2
    initial_mask = create_circular_mask(gal.shape[0], gal.shape[1], radius=initial_rad)
    initial_flux = np.median(gal[initial_mask])

    for step in range(initial_rad + 1, gal.shape[0]//2, 1):
        new_mask = create_circular_mask(gal.shape[0], gal.shape[1], radius=step)
        prev_mask = create_circular_mask(gal.shape[0], gal.shape[1], radius=step-1)
        crust = np.logical_and(new_mask, prev_mask)

        crust_flux = np.median(gal[crust])

        if crust_flux <= initial_flux*0.5:
            return (step - 1)*arcsec_per_pixel

    # No convergence? Return a NaN
    return np.nan

def sgd(fis_0, fis_1, aperture=6, arcsec_per_pixel=0.262):

    gs_0 = []
    rs_0 = []
    zs_0 = []
    sizes_0 = []

    gs_1 = []
    rs_1 = []
    zs_1 = []
    sizes_1 = []

    for fi_0, fi_1 in tqdm(zip(fis_0, fis_1), total=len(fis_0)):
        # 0 == g, 1 == r, 2 == z
        gal_0 = np.load(fi_0)
        gal_1 = np.load(fi_1)

        if np.any(np.isnan(gal_0)) or np.any(np.isnan(gal_1)):
            continue

        g_0 = get_aperture_magnitude(gal_0[0], radius=aperture)
        r_0 = get_aperture_magnitude(gal_0[1], radius=aperture)
        z_0 = get_aperture_magnitude(gal_0[2], radius=aperture)
        size_0 = get_size(gal_0[0], arcsec_per_pixel=arcsec_per_pixel)
        g_1 = get_aperture_magnitude(gal_1[0], radius=aperture)
        r_1 = get_aperture_magnitude(gal_1[1], radius=aperture)
        z_1 = get_aperture_magnitude(gal_1[2], radius=aperture)
        size_1 = get_size(gal_1[0], arcsec_per_pixel=arcsec_per_pixel)

        if np.all(list(map(np.isfinite, 
                       (g_0, r_0, z_0, size_0, g_1, r_1, z_1, size_1)))):
            # We don't want any nans or infs
            gs_0.append(g_0)
            rs_0.append(r_0)
            zs_0.append(z_0)
            size_0.append(size_0)
            gs_1.append(g_1)
            rs_1.append(r_1)
            zs_1.append(z_1)
            size_1.append(size_1)

    gs_0 = to_magnitude(np.array(gs_0))
    rs_0 = to_magnitude(np.array(rs_0))
    zs_0 = to_magnitude(np.array(zs_0))
    size_0 = np.array(size_0)
    gs_1 = to_magnitude(np.array(gs_1))
    rs_1 = to_magnitude(np.array(rs_1))
    zs_1 = to_magnitude(np.array(zs_1))
    size_1 = np.array(size_1)

    emds = []
    pairs = list(zip((gs_0, rs_0, zs_0, size_0, gs_0 - rs_0, rs_0 - zs_0),
                     (gs_1, rs_1, zs_1, size_1, gs_1 - rs_1, rs_1 - zs_1)))

    pairs = [(pair_0[np.logical_and(np.isfinite(pair_0), np.isfinite(pair_1))],
              pair_1[np.logical_and(np.isfinite(pair_0), np.isfinite(pair_1))]) 
              for (pair_0, pair_1) in pairs]

    for pair_0, pair_1 in zip(pairs):
        emd = wasserstein_distance(pair_0, pair_1)
        emds.append(emd)

    names = ["g mag", "r mag", "z mag", "Re", "g - r", "r - z"]
    for emd, name in zip(emds, names):
        print(name, emd)

    print(f"SGD: {np.sum(emds)}")

if __name__ == "__main__":
    # arg parsing
    parser = argparse.ArgumentParser("Calculate the 'Synthetic Galaxy Distance' between two datasets.")
    # Args
    parser.add_argument(type=str, dest=gals_0, help="Glob pointing to a dataset.")
    parser.add_argument(type=str, dest=gals_1, help="Glob pointing to a dataset.")
    args = parser.parse_args()
    
    gals_0 = np.random.permutation(glob(args.gals_0))
    gals_1 = np.random.permutation(glob(args.gals_1))

    sgd(gals_0, gals_1, arcsec_per_pixel=0.262)
