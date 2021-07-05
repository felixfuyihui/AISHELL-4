# AISHELL-4

This project is associated with the recently-released AIHSHELL-4 dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario. The project, served as baseline, is divided into five parts, named ***data_preparation***, ***front_end***, ***asr*** and ***sd***. The Speaker Independent (SI) task only evaluates the ability of front end (FE) and ASR models, while the Speaker Dependent (SD) task evaluates the joint ability of speaker diarization, front end and ASR models. The goal of this project is to simplify the training and evaluation procedure and make it easy and flexible for researchers to carry out experiments and verify neural network based methods.

## Setup

```shell
git clone https://github.com/felixfuyihui/AISHELL-4.git
pip install -r requirements.txt
```
## Introduction

* [Data Preparation](data_preparation): Prepare the training and evaluation data.
* [Front End](front_end): Train and evaluate the front end model. 
* [ASR](asr): Train and evaluate the asr model. 
* [Speaker Diarization](sd): Generate the speaker diarization results. 
* [Evaluation](eval): Evaluate the results of models above and generate the CERs for Speaker Independent and Speaker Dependent tasks respectively.

## General steps
1. Generate training data for fe and asr model and evaluation data for Speaker Independent task.
2. Do speaker diarization to generate rttm which includes vad and speaker diarization information.
3. Generate evaluation data for Speaker Dependent task with the results from step 2.
4. Train FE and ASR model respectively.
5. Generate the FE results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively.
6. Generate the ASR results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively with the results from step 2 and 3 for No FE results.
7. Generate the ASR results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively with the results from step 5 for FE results.
8. Generate CER results for Speaker Independent and Speaker Dependent tasks of (No) FE with the results from step 6 and 7 respectively.




## Citation
If you use this challenge dataset and baseline system in a publication, please cite the following paper:

    @article{fu2021aishell,
             title={AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario},
             author={Fu, Yihui and Cheng, Luyao and Lv, Shubo and Jv, Yukai and Kong, Yuxiang and Chen, Zhuo and Hu, Yanxin and Xie, Lei and Wu, Jian and Bu, Hui and Xin, Xu and Jun, Du and Jingdong Chen},
             year={2021},
             conference={Interspeech2021, Brno, Czech Republic, Aug 30 - Sept 3, 2021}
             }
The paper is available at https://arxiv.org/abs/2104.03603

Dataset is available at http://www.openslr.org/111/ and https://www.myairbridge.com/en/#!/folder/0yo53qiVSCJ4dDlds1r8Mo6fIATsIRnH
    
## Contributors

[<img width="300" height="100" src="https://github.com/felixfuyihui/AISHELL-4/blob/master/fig_aslp.jpg"/>](http://www.nwpu-aslp.org/)[<img width="300" height="100" src="https://github.com/felixfuyihui/AISHELL-4/blob/master/fig_aishell.jpg"/>](http://www.aishelltech.com/sy)
## Code license 

[Apache 2.0](./LICENSE)
