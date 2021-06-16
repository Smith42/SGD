import matplotlib as mpl
mpl.use("Agg")

import numpy as np
from scipy.stats import rankdata, ks_2samp
from glob import glob
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def get_flux(x):
    """
    Approximate flux by taking the sum of pixel values in an image.
    """
    # TODO is there a better way to do this?
    # Convert to mag
    #p10 = np.percentile(x, 10)
    #x = np.where(x > p10, x, 0)
    x = np.max(x)
    print(x)
    return x

def fisher(ps):
    """
    Fisher summation for combining p values.
    """
    return -np.sum(np.log(ps))

def sgd(rgals, sgals, plot=False):
    r_gs = []
    r_rs = []
    r_zs = []
    for fi in tqdm(rgals):
        # 0 == g, 1 == r, 2 == z
        gal = np.load(fi)
        r_gs.append(get_flux(gal[0]))
        r_rs.append(get_flux(gal[1]))
        r_zs.append(get_flux(gal[2]))
    r_gs = np.array(r_gs)
    r_rs = np.array(r_rs)
    r_zs = np.array(r_zs)

    s_gs = []
    s_rs = []
    s_zs = []
    for fi in tqdm(sgals):
        # 0 == g, 1 == r, 2 == z
        gal = np.load(fi)
        s_gs.append(get_flux(gal[0]))
        s_rs.append(get_flux(gal[1]))
        s_zs.append(get_flux(gal[2]))
    s_gs = np.array(s_gs)
    s_rs = np.array(s_rs)
    s_zs = np.array(s_zs)

    ps = []
    pairs = list(zip((r_gs, r_rs, r_zs, r_rs - r_zs),
                     (s_gs, s_rs, s_zs, s_rs - s_zs)))

    if plot:
        f, axs = plt.subplots(1, len(pairs), figsize=(len(pairs)*2, 2), constrained_layout=True)
        
    names = ["g", "r", "z", "r - z"]
    for i, name, pair in zip(range(len(pairs)), names, pairs):
        _, p = ks_2samp(pair[0], pair[1])
        ps.append(p)

        if plot:
            h0 = np.histogram(pair[0], bins=51)
            h1 = np.histogram(pair[1], bins=51)

            axs[i].set_title(name)
            axs[i].bar(h0[1][:-1], h0[0], width=np.diff(h0[1]), align="edge")
            axs[i].bar(h1[1][:-1], h1[0], width=np.diff(h1[1]), align="edge")

    print(ps, fisher(ps[:-1]), fisher(ps))
    print(fisher((1, 0.9, 1)))
    f.savefig("dumped.png", dpi=300)

if __name__ == "__main__":
    shuf = np.random.permutation(glob("data/reals/*.npy"))
    rgals = shuf[:400]
    sgals = shuf[400:800]
    sgals = glob("data/fakes/*.npy")[:400]
    rgals = glob("data/reals/*.npy")[:400]
    print(len(rgals), len(sgals))

    sgd(rgals, sgals, plot=True)
