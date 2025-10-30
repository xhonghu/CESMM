## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.
- For these who failed install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try `cd ctcdecode` and `pip install .`
- Pealse follow [this link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install pytorch geometric
- You can install other required modules by conducting
  `pip install -r requirements.txt`
  `pip install transformers`

## Data Preparation

**Keypoints**
 - [Phoenix-2014](https://drive.google.com/drive/folders/1D_iVtqeARBLO7WcZCTGCAdHXkKqHfF9X?usp=drive_link)
 - [Phoenix-2014T](https://drive.google.com/drive/folders/1XBBqsxJqM4M64iGxhVCNuqUInhaACUwi?usp=drive_link)
 - [CSL-Daily](https://drive.google.com/drive/folders/11AOSOw1tkI78R6OFJv27adikr3OsUFBk?usp=drive_link) 
 
 pre-extracted by HRNet. Please download them and place them under *Data/Phoenix-2014t(Phoenix-2014 or CSL-Daily)*.

Download datasets and extract them, no further data preprocessing needed.

# SLR

### Weights

Here we provide the performance of the model and its corresponding weights.

| Dataset   | Dev WER | Test WER | Pretrained model |
| ---------- | ------- | -------- | --------------------------- |
| Phoenix14  |  19.9  |  19.7  | [[huggingface]](https://huggingface.co/datasets/xhonghu/CESMM/tree/main/work_dirt/phoenix2014) |
| Phoenix14T |  19.0  |  19.4  | [[huggingface]](https://huggingface.co/datasets/xhonghu/CESMM/tree/main/work_dirt/phoenix2014-T) |
| CSL-Daily  |  27.5  |  26.8  | [[huggingface]](https://huggingface.co/datasets/xhonghu/CESMM/tree/main/work_dirt/CSL-Daily) |

### Evaluate (Taking the PT dataset as an example)
Model consensus integration evaluation:
```
python main.py --phase ensemble --dataset phoenix2014
```

Evaluate a single flow model :

```
test_signle:

python main.py --input-type keypoint --dataset phoenix2014 --load-weights ./work_dirt/phoenix2014-T/keypoint/best_model.pt --phase test 

python main.py --input-type bone --dataset phoenix2014 --load-weights ./work_dirt/phoenix2014-T/bone/best_model.pt --phase test

python main.py --input-type keypoint_motion --dataset phoenix2014 --load-weights ./work_dirt/phoenix2014-T/keypoint_motion/best_model.pt --phase test

python main.py --input-type bone_motion --dataset phoenix2014 --load-weights ./work_dirt/phoenix2014-T/bone_motion/best_model.pt --phase test

```


### Training
Train models for the 4 streams separately.

```
python main.py --input-type keypoint --dataset phoenix2014

python main.py --input-type bone --dataset phoenix2014

python main.py --input-type keypoint_motion --dataset phoenix2014

python main.py --input-type bone_motion --dataset phoenix2014
```

### Acknowledgments

Our code is based on [SignGraph](https://github.com/gswycf/SignGraph) and [MSKA](https://github.com/sutwangyan/MSKA).

