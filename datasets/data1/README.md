Dataset Description
-------------------

**trainset.txt:** Training Dataset in format understandable by the model.
**testset.txt:** Test Dataset in format understandable by the model.
**vocabattr.txt:** Dictionary of output labels from the model.
**vocabpos.txt:** Dictionary of POS (Part of Speech) tags used by the model.
**vocabword.txt:** Dictionary of english words understood by the model.
**wordemb-elecrev-100.npy:** 100-dimensional GloVe word embeddings prepared from corpus of electronic reviews, as described in the paper.
**wordemb-text8-100.npy:** 100-dimensional GloVe word embeddings prepared from the Text8 English corpus.
**wordemb-text8-300.npy:** 300-dimensional GloVe word embeddings prepared from the Text8 English corpus.
**raw-tagged-sentences.txt:** Created semi-manually, using patterns, as described in the paper.

Note that the numpy files are big, and hence, not uploaded as part of this repository. They can be downloaded from here: [Dataset](https://zenodo.org/record/1415481#.W5pjkBwScnQ). After download, put them in this directory, for the code to work out-of-the-box, or else, modify the python code slightly, to read those files from the location where you have placed them.