# wordcomp


This code is an implementation of several **semantic composition models**: Wmask, Addmask, Matrix, Fulllex and Lexfunc. Such models can be used to obtain vectorial representations for linguistic units above the word level by combining the representations of the individual words. 

An example application is the composition of compounds: the vectorial representation of 'apple tree' could be obtained by combining the vectorial representations of 'apple' and 'tree'.

More details can be found in the following paper:

Corina Dima: Reverse-engineering Language: A Study on the Semantic Compositionality of German Compounds. In Proceedings of EMNLP 2015, Lisbon, Portugal, pp. pp. 1637–1642
[Download paper: https://aclweb.org/anthology/D/D15/D15-1188.pdf]

## Prerequisites

The code is written in [Lua](http://www.lua.org/about.html) and uses the [Torch scientific computing framework](http://torch.ch/). To run it, you will have to first install Torch and Torch additional packages `nn`, `optim`, `paths` and `xlua`.

## Data


## Training

Training new composition models:

```
$ th script_compose.lua -model Wmask -dataset sample_dataset -embeddings embeddings_set -dim 50
```

The `-model` option specifies which one of the 5 available models should be trained (Wmask, Addmask, Matrix, Fulllex or Lexfunc).

The `-dataset` option specifies which dataset should the model be trained on. The dataset should be available in the `data` folder.

The `-embeddings` option specifies which word representations should be used for training. A model can be trained on the same dataset, but using different word representations (for example embeddings or count vectors with reduced dimensionality). The different representations should be placed in an `embeddings` subfolder in the dataset folder. Each representation should be in its own subfolder.

The `-dim` option specifies the dimensionality of the word representation. (e.g. for 50 dimensions, the script will look in the dataset folder in the specified embeddings subfolder for a file with the same name as the embedding name and the suffix `.50d_cmh.emb`)

For other available options, see the help:
```
$ th script_compose.lua -help
```

# License

MIT
