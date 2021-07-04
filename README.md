# AISHELL-4


The project is divided into five parts, named data_preparation, front_end, asr, sd, respectively. The goal is to simplify the training and evaluation procedure and make it easy and flexible for researchers to do experiments and verify neural network based methods.

## Setup

```shell
git clone https://github.com/felixfuyihui/AISHELL-4.git
pip install -r requirements.txt
```
## Introduction

* [Data Preparation](data_preparation/data_prep.md)
* [Front End](front_end/fe.md)
* [ASR](asr/asr.md)
* [Speaker Diarization](sd/sd.md)
* [Evaluation](eval/eval.md)

## General steps
1. Generate training data for fe and asr model and evaluation data for Speaker Independent task.
2. Do speaker diarization to generate rttm which containing vad and speaker diarization information.
3. Generate evaluation data for Speaker Dependent task with the results of step 2.
4. Train FE and ASR model respectively.
5. Generate the FE results of evaluation data of Speaker Independent and Speaker Dependent tasks respectively.
6. Generate the ASR results of evaluation data of Speaker Independent and Speaker Dependent tasks respectively with the results of step 2 and 3 for No FE results.
7. Generate the ASR results of evaluation data of Speaker Independent and Speaker Dependent tasks respectively with the results of step 5 for FE results.
8. Generate CER results of Speaker Independent and Speaker Dependent tasks without/with FE with the results of step 6 and 7 respectively.



## Citation
If you use this challenge dataset and baseline system in a publication, please cite the following paper:

    @article{fu2021aishell,
             title={AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario},
             author={Fu, Yihui and Cheng, Luyao and Lv, Shubo and Jv, Yukai and Kong, Yuxiang and Chen, Zhuo and Hu, Yanxin and Xie, Lei and Wu, Jian and Bu, Hui and others},
             journal={arXiv preprint arXiv:2104.03603},
             year={2021}
             }
    
## Contributors

[<img width="300" height="100" src="https://github.com/felixfuyihui/AISHELL-4/blob/master/fig_aslp.jpg"/>](https://www.baidu.com/)[<img width="300" height="100" src="https://github.com/felixfuyihui/AISHELL-4/blob/master/fig_aishell.jpg"/>](http://www.aishelltech.com/sy)
## Code license 

[Apache 2.0](./LICENSE)
