```diff
-Important:
```
## 1- Slides employed during the tutorial are already available (check the documents folder).

## 2- We are working on a survey that will compile all the works on weakly and semi-supervised segmentation with deep models, which will be on arxiv within the next weeks.

# Code demo for the weakly supervised segmentation tutorial at MICCAI 2019

![comparison](comparison.gif)

## Requirements
This code was written for Python 3.5+. Most of the required packages include:
```
pytorch (tested with 1.2 and 1.3)
torchvision
numpy
scipy
matplotlib
tqdm
PIL
```
`ImageMagick` (available by default on most Linux distributions) is required for the optional script generating the GIF displayed above.

## Usage
The current version is implemented on python 3.5 and pytorch 0.4.0
```bash
python3 -O weakly_sup_segmentation.py --mode 0
python3 -O weakly_sup_segmentation.py --mode 1
python3 plotResults.py
./gifs.sh
```
The `-O` is optionnal, as it is a switch do disable all the assertions and sanity checks within the code.

## Results
Results are stored in the following folder structure:
```
Results/
    Images/
        Results/Images/Weakly_Sup_CE_Loss/
            yy_Ep_xxxx.png
            ...
        Results/Images/Weakly_Sup_CE_Loss_SizePenalty/
            ...
    Statistics/
        Results/Images/Weakly_Sup_CE_Loss/
            CE_Losses.npy
            ...
        Results/Images/Weakly_Sup_CE_Loss_SizePenalty/
            ...
result.gif
```

## Citations
If you are re-using this code in your research, and the material from this tutorial, please consider citing:

`Hoel Kervadec, Jose Dolz, Meng Tang, Eric Granger, Yuri Boykov, and Ismail Ben Ayed. "Constrained-CNN losses for weakly supervised segmentation." Medical image analysis 54 (2019): 88-99`.

`Ismail Ben Ayed, Christian Desrosiers and Jose Dolz. "Weakly Supervised CNN Segmentation: Models and Optimization."Medical Image Computing and Computer Assisted Intervention − MICCAI 2019 Tutorial`.

## Old code
For reference only, we provide the run that was actually run during the MICCAI tutorial, in the `old_version` folder. This code is compatible with PyTorch 0.4 only, and is actually on old research code that we used at the time.

At the time, we were doing the forward and backward functions manually in NumPy, which complexifies the implementation and verification a bit.
Due to majors update for PyTorch in between, and perhaps few minor differences with the current full PyTorch implementation, small difference in results may arise.

We do not intend to update or support that old version, but users are free to explore it.