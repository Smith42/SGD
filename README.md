## Calculate the 'Synthetic Galaxy Distance' between two galaxy datsets.

This code calculates the 'Synthetic Galaxy Distance' (SGD) described in
'Realistic galaxy simulation via score-based generative models'.

The metric is the sum of Wasserstein-1 distances between emergent physical
galaxy properties from two given galaxy postage stamp observation datasets.
Here we compare size and flux distributions.

The SGD returns a single number, where a lower value denotes a closer
match between two datasets. When combined with the [Fréchet Inception
Distance](https://github.com/mseitzer/pytorch-fid) for visual and
morphological similarity, we gain a good overview of the similarity
between two large galaxy photometry datasets.

### Usage

To run the code clone this repo and execute

```bash
python sgd.py path/to/dataset1 path/to/dataset2 
```

### Citing

If you find this work useful please consider citing:

```bibtex
@article{smith2021,
    title={Realistic galaxy image simulation via score-based generative models},
    author={Michael J. Smith and James E. Geach and Ryan A. Jackson and Nikhil Arora and Connor Stone and St{\'{e}}ephane Courteau},
    journal = {arXiv e-prints},
    year={2021},
    eprint = {2111.01713}
}
```
