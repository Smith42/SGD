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
    p10 = np.percentile(x, 10)
    x = np.where(x > p10, x, 0)
    #x = 22.5 - 2.5*np.log10(x)
    return np.sum(x)

def fisher(ps):
    """
    Fisher summation for combining p values.
    """
    return -np.sum(np.log(ps))

def sgd(rgals, sgals):
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
    s_zs = np.array(s_gs)

    ps = []
    for pair in zip((r_gs, r_rs, r_zs, r_rs - r_zs), 
                    (s_gs, s_rs, s_zs, s_rs - s_zs)):
        _, p = ks_2samp(pair[0], pair[1])
        ps.append(p)

    print(ps, fisher(ps))
    print(fisher((1, 0.9, 1)))

    #plt.plot(to_compare[0], to_compare[1], marker)
    #plt.savefig("ims/colour-colour.png")

if __name__ == "__main__":
    rgals = glob("data/reals/*.npy")[:500]
    sgals = glob("data/fakes/*.npy")[:500]

    sgd(rgals, sgals)
