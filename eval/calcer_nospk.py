#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
import os
import wave
import re
import argparse
import textgrid
import linecache
import numpy as np
import random
import soundfile as sf
import re
import itertools
from itertools import product
from subprocess import call
import pandas as pd
import re

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def splitdecode_intosession(args):
    fin = args.asr_decode_time_format
    flag = 'L_R003S01C02'
    if not os.path.exists('./ctm_'+args.whether_fe):
        os.mkdir('./ctm_'+args.whether_fe)
    fout = open('./ctm_'+args.whether_fe+'/'+flag+'.ctm','w')
    for i in range(len(open(fin,'r').readlines())):
        line = get_line_context(fin, i+1)
        wavid = line.split(' ')[0]
        if wavid == flag:
            fout.write(line+'\n')
            fout.flush()
        else:
            flag = wavid
            fout = open('./ctm_'+args.whether_fe+'/'+flag+'.ctm','w')
            fout.write(line+'\n')
            fout.flush()

def calcer_nospk(args):
    ctmlist = os.listdir('./ctm_'+args.whether_fe)
    stmlist = os.listdir('./stm/')
    if not os.path.exists('ctmraw_'+args.whether_fe):
        os.mkdir('ctmraw_'+args.whether_fe)
    ctmlist = sorted(ctmlist)
    stmlist = sorted(stmlist)
    for i in range(len(stmlist)):
        ctm = os.path.join('ctm_'+args.whether_fe, ctmlist[i])
        stm = os.path.join('stm',stmlist[i])
        wavid = ctm.split('/')[-1]
        wavid = wavid.split('.ctm')[0]
        outpath = './ctmraw_'+args.whether_fe
        call(['./asclite','-r',stm,'stm','-h',ctm,'ctm','-o','rsum','-O',outpath])
    wrd = 0
    err = 0
    ctmrawlist = os.listdir('./ctmraw_'+args.whether_fe)
    ctmrawlist = sorted(ctmrawlist)
    for i in range(len(ctmrawlist)):
        ctmsys = os.path.join('ctmraw_'+args.whether_fe,ctmrawlist[i])
        for j in range(50):
            ctmline1 = get_line_context(ctmsys, j+1)
            ctmline2 = ctmline1.split(' ')
            if len(ctmline2) > 2:
                if ctmline2[1] == 'Sum':
                    ctmline = ctmline1
                    break
        ptrn = re.compile('\|\s*Sum\s*\|\s*\d+\s+(\d+)\s*\|(.+)\|.+\|$')
        m = ptrn.match(ctmline)
        gp1 = m.group(1).strip().split()
        gp2 = m.group(2).strip().split()
        wrd = float(gp1[0]) + wrd
        err = float(gp2[-2]) + err
    print("CER is:", err/wrd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("asr_decode_time_format",
                        type=str,
                        help="wav_list",
                        default="")
    parser.add_argument("whether_fe",
                        type=str,
                        help="whether frontend (nofe/fe)",
                        default="nofe")
    args = parser.parse_args()
    splitdecode_intosession(args)
    calcer_nospk(args)