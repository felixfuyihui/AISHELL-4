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

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()

def run(args):
    wav_list = args.wav_list
    rttmlist = os.listdir('./rttm_sd/')
    rttmlist = sorted(rttmlist)
    output_dir = args.output_dir + '/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range((len(open(wav_list,'r').readlines()))):
        wavpath = get_line_context(wav_list, i + 1) ##wav的列表
        wavid = wavpath.split('/')[-1]
        wavid = wavid.split('.wav')[0]
        y, fs = sf.read(wavpath)
        if not os.path.exists(output_dir+'/'+wavid):
            os.makedirs(output_dir+'/'+wavid)


        startlist = []
        endlist = []
        spkidlist = []
        for j in range(len(open('./rttm_sd/'+rttmlist[i],'r').readlines())):
            line = get_line_context('./rttm_sd/'+rttmlist[i], j + 1)
            info = line.split(' ')
            start = float(info[3])
            end = float(info[3])+float(info[4])
            start = '%.2f' % start
            end = '%.2f' % end
            startlist.append(start)
            endlist.append(end)
            spkidlist.append(info[7])
        for j in range(len(startlist)):
            cut = y[int(float(startlist[j])*16000):int(float(startlist[j])+float(endlist[j])*16000)]
            start = str(int(float(startlist[j])*100))
            end = str(int(float(endlist[j])*100))
            sf.write(output_dir+'/'+wavid+'/'+wavid+'-'+start.rjust(6,'0')+'-'+end.rjust(6,'0')+'-'+spkidlist[j]+'.wav',cut,fs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_list",
                        type=str,
                        help="wav_list",
                        default="rawwav_list/test/wav.txt")
    parser.add_argument("--output_dir",
                        type=str,
                        help="output_dir",
                        default="data_spk")

    args = parser.parse_args()
    run(args)