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
    if not os.path.exists('./rttm_'+args.whether_fe):
        os.mkdir('./rttm_'+args.whether_fe)
    fout = open('./rttm_'+args.whether_fe+'/'+flag+'.ctm','w')
    for i in range(len(open(fin,'r').readlines())):
        line = get_line_context(fin, i+1)
        wavid = line.split(' ')[0]
        if wavid == flag:
            fout.write(line+'\n')
            fout.flush()
        else:
            flag = wavid
            fout = open('./rttm_'+args.whether_fe+'/'+flag+'.ctm','w')
            fout.write(line+'\n')
            fout.flush()

    rttmlist = os.listdir('./rttm_sd/')
    rttmlist = sorted(rttmlist)
    ctmlist = os.listdir('./rttm_'+args.whether_fe)
    ctmlist = sorted(ctmlist)

    for i in range(len(ctmlist)):
        rttm = './rttm_sd/'+rttmlist[i]
        ctm = './rttm_'+args.whether_fe+'/'+ctmlist[i]
        rttmspkidlist = []
        rttmwavidlist = []
        rttmstartlist = []
        rttmbiaslist = []
        for j in range(len(open(rttm,'r').readlines())):
            diaresult = get_line_context(rttm, j + 1)
            diaresultsplit = diaresult.split(' ')
            wavid, start, bias, spkid = diaresultsplit[1], diaresultsplit[3], diaresultsplit[4], diaresultsplit[7]
            start = float(start)
            bias = float(bias)
            start = float('%.2f' % start)
            bias = float('%.2f' % bias)
            rttmwavidlist.append(wavid)
            rttmstartlist.append(start)
            rttmbiaslist.append(bias)
            rttmspkidlist.append(spkid)

        ctmwavidlist = []
        ctmstartlist = []
        ctmbiaslist = []
        ctmtokenlist = []
        ctmspkidlist = []
        for j in range(len(open(ctm,'r').readlines())):
            decoderesult = get_line_context(ctm, j + 1)
            decoderesultsplit = decoderesult.split(' ')

            wavid, start, bias, token = decoderesultsplit[0], decoderesultsplit[2], decoderesultsplit[3], decoderesultsplit[4]
            start = float(start)
            bias = float(bias)
            start = float('%.2f' % start)
            bias = float('%.2f' % bias)

            ctmwavidlist.append(wavid)
            ctmstartlist.append(start)
            ctmbiaslist.append(bias)
            ctmtokenlist.append(token)
        sumbiasctm = 0.0
        flagrttm = 0
        ctmsegpointlist = []

        for j in range(len(ctmstartlist)):
            if abs(ctmstartlist[j] - rttmstartlist[flagrttm]) < 0.12: #0.12
                ctmsegpointlist.append(j)
                flagrttm = flagrttm + 1
            if abs(ctmstartlist[j] - rttmstartlist[-1]) < 0.12:
                break
        ctmsegpointlist.append(len(open(ctm,'r').readlines()))
        outrttm = open('./rttm_'+args.whether_fe+'/'+str(rttmwavidlist[0])+'.rttm','w')
        spknum = list(set(rttmspkidlist))
        for j in range(len(spknum)):
            outrttm.write('SPKR-INFO ' + wavid +' 0 <NA> <NA> <NA> UNKNOWN '+str(spknum[j])+'\n')
        for j in range(len(ctmsegpointlist)-1):
            outrttm.write('SPEAKER ' + ctmwavidlist[0] + ' 0 ' + str(rttmstartlist[j]) + ' ' + str(rttmbiaslist[j]) + ' <NA> <NA> ' +str(rttmspkidlist[j])+'\n')
            for k in range(int(ctmsegpointlist[j]), int(ctmsegpointlist[j+1])):
                outrttm.write('LEXEME '+ctmwavidlist[0] + ' 0 ' + str(ctmstartlist[k])+' ' + str(ctmbiaslist[k])+' ' + ctmtokenlist[k]+ ' LEX ' + str(rttmspkidlist[j])+'\n')

        os.remove(ctm)  


def calcer_spk(args):

    rttmlist = os.listdir('./rttm_'+args.whether_fe)
    csvlist = os.listdir('./csv/')
    stmlist = os.listdir('./stm/')
    rttmlist = sorted(rttmlist)
    csvlist = sorted(csvlist)
    stmlist = sorted(stmlist)
    if not os.path.exists('./rttmraw_'+args.whether_fe):
        os.mkdir('./rttmraw_'+args.whether_fe)


    for i in range(len(stmlist)):
        rttm = './rttm_'+args.whether_fe + '/' + rttmlist[i]
        stm = './stm/' + stmlist[i]
        csv = './csv/' + csvlist[i]
        outpath = './rttmraw_'+args.whether_fe
        call(['./asclite','-r',stm,'stm','-h',rttm,'rttm','-spkr-align',csv,'-o','rsum','-O',outpath])
        
    wrd = 0
    err = 0
    ctmrawlist = os.listdir('./rttmraw_'+args.whether_fe)
    # ctmrawlist = os.listdir('/home/environment/yhfu/aishell4_release/test_stm_rttm_csv/1')
    ctmrawlist = sorted(ctmrawlist)
    for i in range(len(ctmrawlist)):
        ctmsys = './rttmraw_'+args.whether_fe + '/' + ctmrawlist[i]
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
    calcer_spk(args)