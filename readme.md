# MICCAI 2022 Tutorial
## Weakly Supervised CNN Segmentation: Models and Optimization

**The code for the 2021 tutorial is available at commit [c8d5d248dd8b504b53c56449ea526799df87e3ea](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial/commit/c8d5d248dd8b504b53c56449ea526799df87e3ea) (tag `MICCAI2021`).** Other older tutorials (`MICCAI2020`, `MICCAI2019`) are also tagged.

This repository contains the code of the hand-on tutorial. The hands-on will be done in three main parts:
* **naive sizeloss**, to introduce the pipeline and general methodology;
* **combined size and centroid supervision**, with a quadratic penalty;
* combined size and centroid supervision, **with an extended log-barrier**.

### Hands-on
![preview.gif](preview.gif)

The first goal is to enforce some inequality constraints on the size of the predicted segmentation in the form:
```
lower bound <= predicted size <= upper bound
```
where `predicted size` is the sum of all predicted probabilities (softmax) over the whole image.

To make the example simpler, we will define the lower and upper bounds to 0.9 and 1.1 times the ground truth size. All the code is contained within the `code/` folder. **The following assume you moved in that directory.**

#### Requirements
The code has those following dependencies:
```
python3.7+
pytorch (latest)
torchvision
numpy
tqdm
```
Running the PROMISE12 example requires some additional packages:
```
simpleitk
scikit-image
PIL
```
ACDC relies on, for the slicing:
```
nibabel
```

#### Data
The data for the toy example is stored in `code/data/TOY`. If you wish, you can regenerate the dataset with:
```
make -B data/TOY
make -B data/TOY2
```
or you can use [gen_toy.py](code/gen_toy.py) directly.

Participants willing to try the PROMISE12 setting need to download the data themselves, then put the .zip inside the `code/data` folder (a list of files is available in `code/data/promise12.lineage`). Once the three files are there, the slicing into 2D png files is automated:
```
make data/PROMISE12
```
It will:
* checks data integrity
* extract the zip
* slice into 2d slices
* generate weak labels from the actual ground truth

The same goes for ACDC:
```
make data/ACDC
```

#### Training
```
>>> ./main.py -h
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY,PROMISE12}] [--mode {constrained,unconstrained,full}] [--gpu]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY,PROMISE12,ACDC}
  --mode {constrained,unconstrained,full}
  --gpu
```
The toy example is designed to run under 5 minutes on a laptop, training on CPU. The following commands are equivalent
```
python3 main.py
./main.py
./main.py --epochs 200 --dataset TOY --mode unconstrained
```

The three modes correspond to:
* unconstrained: use the weak labels, with only a partial cross-entropy (won't learn anything)
* constrained: use the weak labels, with partial cross-entropy + size constraint (will learn)
* full: use full labels, with cross entropy (will learn, for obvious reasons)

The settings for PROMISE12 are too simple to get state of the art results, even in the `full` mode, but it gives a good starting point for new practitioners to then build on.

Examples constraining both the size and the centroid, without resorting to any pixel-wise supervision, are shown:
```
>>>  ./main_centroid.py -h
usage: main_centroid.py [-h] [--epochs EPOCHS] [--dataset {TOY2}] [--mode {quadratic,logbarrier}] [--gpu]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2}
  --mode {quadratic,logbarrier}
  --gpu
```

## MICCAI 2021 recordings
While the recording and slides are not yet available, the ones from last year are still online.

### Slides
Slides from the three sessions are available in the [`slides/`](slides/) folder.

### Recordings
* [Session 1](https://drive.google.com/file/d/1NVn2J4y6l7_Yxw6RGBD2CEIEedliccjQ/view?usp=sharing): Structure-driven priors: _Regularization_
* [Session 2](https://drive.google.com/file/d/1wAVxBk4U45-SZhDWviCgFShytf0wrJze/view?usp=sharing): Knowledge-driven priors (e.g., anatomy): _Constraints_
* [Session 3](https://drive.google.com/file/d/1EohLWWa5vMmEMxw3Rqk4eYaDzbr_Clp2/view?usp=sharing): Data-driven priors: _Adversarial learning_
* [Session 4](https://drive.google.com/file/d/1NMU7z0KhXYX6idgCBehdaNVAifOE6Ey3/view?usp=sharing): Hands-on: _Size constraints_