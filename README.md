# Movie Screenplay Parser

This repository contains the data and code used to train a movie screenplay parser.
We have released the parser for you to use in your projects.

To use the parser, first create the following conda environment (or use your favorite environment manager)

```shell
# create conda environment
conda create -n parser python=3.8

# install spacy (gpu-optimized)
conda install -c conda-forge spacy cupy

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install sentence-transformers library
conda install -c conda-forge sentence-transformers
```

Please refer to the install pages of pytorch, spacy, and sentence-transformers to get te most up-to-date instructions.

I have validated my parser with python 3.8, pytorch 1.7, and sentence-transformers 2.2.
Other versions should work out equally well.

Next, look at the following example to understand how you can instantiate a screenplay parser and use it to parse a
movie script.

```python
from screenplayparser import ScreenplayParser

# instantiate a transformer-based parser by setting use_rules=False
# device_id is the GPU id the parser will use
trx_parser = ScreenplayParser(use_rules=False, device_id=1)

# instantiate a rule-based parser by setting use_rules=True
rule_parser = ScreenplayParser(use_rules=True)

# read a script and save it as a list of strings
# SCRIPT_PATH is the filepath of the movie script
with open(SCRIPT_PATH) as reader:
    script = reader.read().split("\n")

# trx_tags contains the tag per script line found by the transformer-based parser
# rule_tags contains the tag per script line found by the rule-based parser
trx_tags = trx_parser.parse(script)
rule_tags = rule_parser.parse(script)
```

The transformer-based parser is much slower (100x), but more accurate than the rule-based parser.