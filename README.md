# music-reverse-recommendation

***Please open the notebook on nbviewer by clicking on the badge below or on a jupyter server for the dynamic html to render properly***.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/tkalim/music-reverse-recommendation/blob/main/reverse-recommend.ipynb)

Reverse music recommendation on the [last.fm dataset](https://www.kaggle.com/neferfufi/lastfm?select=userid-timestamp-artid-artname-traid-traname.tsv) using Matrix Factorization.

to reproduce environment:

- with conda:
    `conda env create --file environment.yml`
- with pip:
`pip install -r requirements.txt`

If using jupyterlab execute the following for the tqdm bar to render properly:

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`
