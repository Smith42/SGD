import matplotlib as mpl
mpl.use("Agg")

import numpy as np
from scipy.stats import rankdata, ks_2samp
from glob import glob
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

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

def get_aperture_magnitude(x, radius=10):
    """
    Approximate flux by taking the sum of pixel values in an image.
    """
    mask = create_circular_mask(x.shape[0], x.shape[1], radius=radius)
    x = np.sum(x[mask])
    return x

def get_size(x):
    """
    Get half-light radius of galaxy, assuming that centre of image == centre of galaxy.
    """
    initial_mask = create_circular_mask(x.shape[0], x.shape[1], radius=10)
    initial_flux = np.median(x[initial_mask])

    for step in range(12, x.shape[0], 2):
        new_mask = create_circular_mask(x.shape[0], x.shape[1], radius=step)
        prev_mask = create_circular_mask(x.shape[0], x.shape[1], radius=step-2)
        crust = np.logical_and(new_mask, prev_mask)

        crust_flux = np.median(x[crust])

        if crust_flux <= initial_flux*0.5:
            return (step - 1)*0.262

    return None

def fisher(ps):
    """
    Fisher summation for combining p values.
    """
    return -np.sum(np.log(ps))

def sgd(rgals, sgals, plot=False):
    r_gs = []
    r_rs = []
    r_zs = []
    r_size = []
    for fi in tqdm(rgals):
        # 0 == g, 1 == r, 2 == z
        gal = np.load(fi)
        if np.any(np.isnan(gal)):
            continue
        galsize = get_size(gal[0])
        if galsize is not None:
            r_size.append(galsize)

        r_gs.append(get_aperture_magnitude(gal[0]))
        r_rs.append(get_aperture_magnitude(gal[1]))
        r_zs.append(get_aperture_magnitude(gal[2]))

    r_gs = np.array(r_gs)
    r_rs = np.array(r_rs)
    r_zs = np.array(r_zs)

    s_gs = []
    s_rs = []
    s_zs = []
    s_size = []
    for fi in tqdm(sgals):
        # 0 == g, 1 == r, 2 == z
        gal = np.load(fi)
        if np.any(np.isnan(gal)):
            continue
        galsize = get_size(gal[0])
        if galsize is not None:
            s_size.append(galsize)

        s_gs.append(get_aperture_magnitude(gal[0]))
        s_rs.append(get_aperture_magnitude(gal[1]))
        s_zs.append(get_aperture_magnitude(gal[2]))

    s_gs = np.array(s_gs)
    s_rs = np.array(s_rs)
    s_zs = np.array(s_zs)

    ps = []
    pairs = list(zip((r_gs, r_rs, r_zs, r_size, r_gs - r_rs, r_rs - r_zs),
                     (s_gs, s_rs, s_zs, s_size, s_gs - s_rs, s_rs - s_zs)))

    if plot:
        f, axs = plt.subplots(2, len(pairs)//2, figsize=(len(pairs), 4), constrained_layout=True)

    names = ["g", "r", "z", "size", "g - r", "r - z"]
    for i, name, pair in zip(range(len(pairs)), names, pairs):
        _, p = ks_2samp(pair[0], pair[1])
        ps.append(p)
        print(np.min(pair[0]), np.min(pair[1]), np.max(pair[0]), np.max(pair[1]))

        if plot:
            bins = np.linspace(np.min(pair[1]), np.max(pair[1]), 51)
            h0 = np.histogram(pair[0], bins=bins)
            h1 = np.histogram(pair[1], bins=bins)

            axs.ravel()[i].set_title(name)
            axs.ravel()[i].bar(h0[1][:-1], h0[0], width=np.diff(h0[1]), align="edge", alpha=0.5, label="r")
            axs.ravel()[i].bar(h1[1][:-1], h1[0], width=np.diff(h1[1]), align="edge", alpha=0.5, label="s")

    print(ps)
    print(fisher(ps))
    axs.ravel()[-1].legend()
    f.savefig("dumped.png", dpi=300)

if __name__ == "__main__":
    shuf = np.random.permutation(glob("data/reals/*.npy"))
    rgals = shuf[:200]
    sgals = shuf[200:400]
    #sgals = glob("data/reals/*.npy")[:400]
    #rgals = glob("data/fixed_fakes/*.npy")[:400]
    print(len(rgals), len(sgals))

    sgd(rgals, sgals, plot=True)
