Robust-EQA
======

Code for the paper

Robust-EQA: Robust Learning for Embodied Question Answering with Noisy Labels
---------
Haonan Luo, Guosheng Lin


Setup
#

Download the [SUNCG](https://github.com/facebookresearch/House3D/blob/master/INSTRUCTION.md#usage-instructions) dataset, [House3D](https://github.com/abhshkdz/House3D/tree/master/renderer#rendering-code-of-house3d) virtual environment and install [EmbodiedQA](https://github.com/facebookresearch/EmbodiedQA/blob/master/README.md) framework.

Replace all files in the training folder into the project of EmboiedQA.

Robust-VQA
----

For low-level noisy environments (20\% of noisy labels):

For extremely noisy environments (45\% of noisy labels):


Robust-Navigation
----

For low-level noisy environments (20\% of noisy labels):

For extremely noisy environments (45\% of noisy labels):

Joint-Robust learning
---
python train_eqa.py -nav_checkpoint_path /path/to/nav/ques-image-pacman/checkpoint.pt -ans_checkpoint_path /path/to/vqa/ques-image/checkpoint.pt -identifier ques-image-eqa -log


