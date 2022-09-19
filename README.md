# Approximating The Pareto Front Of Style Transfer Using Hypernetworks

## Install

```bash
git clone https://github.com/nooblearner21/pareto-style-transfer.git
cd pareto-style-transfer
pip install -r requirements.txt
```

## Running Compressed and Large models
After training a new model you need to use the same chunking configuration you used during the training.

Currently the default values are suitable for the provided large models so everything runs out of the box
If you wish to use the compressed model you need to use the configuration that was used to train them(300 chunks of 2800x2 matrices).

The configuration can be provided instantly to the CLI - Example is provided below which also works out of the box for compressed models!

## Rays
Passing the ray argument determines the style-content ratio. The easiest way is to treat them as a two-dimensional simplex
for example --ray 0.2 0.8 can intuitevly seen as a 20% content and 80% style content


## Stylize example with our Kadinsky's model

```bash
python pareto/main.py --stylize --image pareto/eiffel.jpg --stylize-model-path examples/Kadinsky/kadinsky_ours.pth --ray 0.1 0.9
```

## Stylize example with our Compressed Kadinsky's model

```bash
python pareto/main.py --stylize --image pareto/eiffel.jpg --stylize-model-path examples/Kadinsky/kadinsky_ours_compressed.pth --hypervec-dim 2800 --num-hypervecs 2 --chunks 300 --ray 0.1 0.9
```

## Training example

```bash
python pareto/main.py --train --image examples/Kadinsky/kadinsky.jpg --train-data-path dataset
```

We added a dataset folder with 4 samples that you can use to train the network and see the parameter tunning and losses,
In order to fully train the network we used the MS-COCO 2014 val dataset.

# pareto-hypernetworks-style-transfer
