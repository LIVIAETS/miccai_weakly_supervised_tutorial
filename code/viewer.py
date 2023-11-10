#!/usr/bin/env python3.11

# MIT License

# Copyright (c) 2023 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.colors import ListedColormap


# Based on torchvision, itself based on
# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                         'has_instances', 'ignore_in_eval', 'color'])

city_classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),

        CityscapesClass('gta5thing', 34, 34, 'void', 0, False, True, (255, 255, 255))
        # CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


def extract(pattern: str, string: str) -> Optional[str]:
        if m_ := re.match(pattern, string):
                return m_.group(1)

        return None  # ID not found


def display_item(axe, img: np.ndarray, mask: np.ndarray | None, contour: bool, cmap,
                 args):
        if mask is not None:
                if mask.shape != img.shape[:2]:
                        assert mask.dtype == np.uint8
                        m = np.asarray(Image.fromarray(mask).resize(img.shape[:2],
                                                                    resample=Image.Resampling.NEAREST)).astype(np.uint8)
                        assert set(np.unique(mask)) == set(np.unique(m))
                else:
                        m = mask

                assert m.dtype == np.uint8
                assert set(np.unique(m)) <= set(range(args.C)), (np.unique(m),
                                                                 np.unique(mask),
                                                                 args.C,
                                                                 m.dtype,
                                                                 mask.dtype)

        axe.imshow(img, cmap="gray")

        if mask is not None:
                if contour:
                        axe.contour(m, cmap=cmap, interpolation='none')
                else:
                        axe.imshow(m, cmap=cmap, interpolation='none', alpha=args.alpha, vmin=0, vmax=args.C)
        axe.axis('off')


def display(background_names: list[str], segmentation_names: list[list[str]],
            indexes: list[int], column_title: list[str], row_title: list[str],
            crop: int, contour: bool, remap: dict, fig=None, args=None) -> None:
        if not fig:
                fig = plt.figure()
        grid = gridspec.GridSpec(len(indexes) + args.legend, len(segmentation_names),
                                 height_ratios=[((0.9 + 0.1 * ~args.legend) / len(indexes))
                                                for _ in range(len(indexes))] + ([0.1] if args.legend else []))
        grid.update(wspace=0.025, hspace=0.05)

        names: list[str]
        if args.cmap == 'cityscape':
                colors = [tuple(c / 255 for c in e.color) for e in city_classes]
                names = [e.name for e in city_classes]

                cmap = ListedColormap(colors, 'cityscape')
        else:
                # cmap = args.cmap
                cmap = matplotlib.cm.get_cmap(args.cmap)
                names = list(map(str, range(args.C))) if not args.class_names else args.class_names
                assert len(names) == args.C

        if args.legend:
                ax = plt.subplot(grid[-1, :])

                ax.bar(list(range(args.C)), [1] * args.C,
                       tick_label=names,
                       color=[cmap(v / args.C) for v in range(args.C)])

                ax.set_xticklabels(names, rotation=60)
                ax.set_xlim([-0.5, args.C - 0.5])
                ax.get_yaxis().set_visible(False)
                ax.set_title("Legend")

        for i, idx in enumerate(indexes):
                # img: np.ndarray = imread(background_names[idx])
                img: np.ndarray = np.asarray(Image.open(background_names[idx]))

                if crop > 0:
                        img = img[crop:-crop, crop:-crop]

                for j, names in enumerate(segmentation_names):
                        ax_id = len(segmentation_names) * i + j
                        # axe = grid[ax_id]
                        axe = plt.subplot(grid[ax_id])

                        name: str | None = names[idx]
                        seg: np.ndarray | None
                        if name:
                                seg = np.asarray(Image.open(name))
                                if crop > 0:
                                        seg = seg[crop:-crop, crop:-crop]  # type: ignore
                                if remap:
                                        for k, v in remap.items():
                                                seg[seg == k] = v  # type: ignore
                        else:
                                seg = None

                        display_item(axe, img, seg, contour, cmap, args)

                        if j == 0:
                                print(row_title[idx])
                                axe.text(-30, img.shape[1] // 2, row_title[idx], rotation=90,
                                         verticalalignment='center', fontsize=14)
                        if i == 0:
                                axe.set_title(column_title[j])

                fig.tight_layout()


def get_image_lists(img_source: str, folders: list[str], id_regex: str) -> tuple[list[str], list[list[str]], list[str]]:
        '''
        We assume here that all files already exists. I could simply get the filelist from img_source, and load then
        dynamically.
        But I prefer to make sure, for now, that all folders contain exactly the same files (same number, same names
        no additions, no substraction).
        '''
        path_source: Path = Path(img_source)
        raw_background_names: list[str] = sorted(map(str, path_source.glob("*")))
        raw_segmentation_names: list[list[str]] = [sorted(map(str, Path(folder).glob("*"))) for folder in folders]

        extracter: Callable[[str], Optional[str]] = partial(extract,
                                                            id_regex)

        background_names: list[str]
        segmentation_names: list[list[str]]
        ids: list[str]

        background_names = [bg for bg in raw_background_names if extracter(bg) is not None]
        segmentation_names = [[sn for sn in sl if extracter(sn) is not None]
                              for sl in raw_segmentation_names]

        ids = [e for e in map(extracter, background_names) if e is not None]

        for names, folder in zip(segmentation_names, folders):
                folder_ids: list[str] = [e for e in map(extracter, names) if e is not None]

                try:
                        assert len(background_names) == len(names)
                        assert ids == folder_ids
                except AssertionError:
                        print(f"Error verifying content for folder {folder}")
                        print(f"Background folder '{img_source}': {len(background_names)} imgs")
                        # pprint(background_names[:10])
                        print(f"Folder '{folder}': {len(names)} imgs")
                        # pprint(names[:10])

                        s_ids: set[str] = set(ids)
                        s_f_ids: set[str] = set(folder_ids)
                        diff: set[str] = (s_ids - s_f_ids) | (s_f_ids - s_ids)
                        print(f"Difference between the two sets: ({len(diff)} images)")
                        pprint(diff)

                        raise

        return background_names, segmentation_names, ids


class EventHandler(object):
        def __init__(self, order: list[int], n: int, draw_function: Callable, fig):
                self.order: list[int] = order
                self.draw_function: Callable = draw_function
                self.n = n
                self.i = 0
                self.fig = fig

        def __call__(self, event):
                if event.button == 1:  # next
                        self.i += 1
                elif event.button == 3:  # prev
                        self.i -= 1

                a = self.i * self.n

                self.redraw(a)

        def redraw(self, a):
                self.fig.clear()

                idx: list[int] = self.order[a:a + self.n]

                self.draw_function(idx, fig=self.fig)

                self.fig.canvas.draw()


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Display the requested data.")
        parser.add_argument("--img_source", type=str, required=True,
                            help="The folder containing the images (display background).")

        parser.add_argument("-n", type=int, default=3,
                            help="The number of images to sample per window.")
        parser.add_argument("--seed", type=int, default=0,
                            help="The seed for the number generator. Used to sample the images. \
                                                         Useful to reproduce the same outputs between runs.")
        parser.add_argument("--crop", type=int, default=0,
                            help="The number of pixels to remove from each border.")
        parser.add_argument("-C", type=int, default=2,
                            help="""Number of classes. Useful when not all of them appear on each images.
(e.g., 5 classes segmentation but samples contains only classes 0 1 3.)""")

        parser.add_argument("--alpha", default=0.5, type=float)

        parser.add_argument("--id_regex", type=str, default=".*/(.*).png",
                            help="""The regex to extract the image id from the images names.
Required to match the images between them.
Can easily be modified to also handle .jpg""")
        parser.add_argument("folders", type=str, nargs='*',
                            help="The folder containing the source segmentations.")
        parser.add_argument("--display_names", type=str, nargs='*',
                            help="""The display name for the folders in the viewer.
If not set, will use the whole folder name.""")
        parser.add_argument("--class_names", type=str, nargs='*',
                            help="Give names to classes, useful for multi-organs segmentation.")
        parser.add_argument("--remap", type=str, default="{}",
                            help="Remap some mask values if needed. Useful to suppress some classes.")

        parser.add_argument("--no_contour", action="store_true",
                            help="Do not draw a contour but a transparent overlap instead.")
        parser.add_argument("--legend", action="store_true",
                            help="When set, display the legend of the colors at the bottom")
        parser.add_argument("--add_empty_column", "--show_img", action="store_true",
                            help="Add an empty column on the left, with only the image.")

        parser.add_argument("--cmap", default='rainbow', choices=list(plt.colormaps()) + ['cityscape'])
        args = parser.parse_args()

        return args


def main() -> None:
        args: argparse.Namespace = get_args()
        np.random.seed(args.seed)
        mpl.rcParams['image.interpolation'] = 'none'

        background_names: list[str]
        segmentation_names: list[list[str]]
        ids: list[str]
        background_names, segmentation_names, ids = get_image_lists(args.img_source, args.folders, args.id_regex)

        if args.display_names is None:
                display_names = [f for f in args.folders]
        else:
                assert len(args.display_names) == len(args.folders), (args.display_names, args.folders)
                display_names = args.display_names

        if args.add_empty_column:
                segmentation_names = [[""] * len(segmentation_names[0])] + segmentation_names
                display_names = ["Image"] + display_names

        order: list[int] = list(range(len(background_names)))
        order = np.random.permutation(order)  # type: ignore

        draw_function = partial(display, background_names, segmentation_names,
                                column_title=display_names,
                                row_title=ids,
                                crop=args.crop,
                                contour=not args.no_contour,
                                remap=eval(args.remap),
                                args=args)

        fig = plt.figure()
        event_handler = EventHandler(order, args.n, draw_function, fig)
        fig.canvas.mpl_connect('button_press_event', event_handler)

        draw_function(order[:args.n], fig=fig)
        plt.show()


if __name__ == "__main__":
        main()
