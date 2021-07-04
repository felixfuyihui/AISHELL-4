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
    textgrid_list = args.textgrid_list
    output_dir = args.output_dir + '/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for textgrid_num in range(len(open(textgrid_list,'r').readlines())):
        grid = get_line_context(textgrid_list, textgrid_num + 1) #textgrid的列表
        tgridobj = textgrid.TextGrid()
        tgridobj.read(grid)
        n_spk = len(tgridobj)
        conferencetime = tgridobj.maxTime
        conferencetime = float(conferencetime)
        conferencetime = '%.2f' % conferencetime
        conferencetime = float(conferencetime)
        timetable =  np.zeros([n_spk, int(conferencetime*100)]) #建立一个timetable来看哪些点上没有说话人出现，然后用来切句
        for spk in range(n_spk):
            n_iterval = len(tgridobj[spk])
            for j in range(n_iterval):
                iterval = tgridobj[spk][j]
                if iterval.mark != '' and iterval.mark != ' ' and iterval.mark != '<%>' and iterval.mark != '<$>':
                    minTime = iterval.minTime
                    maxTime = iterval.maxTime
                    minTime = float(minTime)
                    maxTime = float(maxTime)
                    minTime = '%.2f' % minTime
                    maxTime = '%.2f' % maxTime
                    minTime = float(minTime)
                    maxTime = float(maxTime)
                    timetable[spk, int(minTime*100):int(maxTime*100)] = 1
        flagcut = 0
        positionstart = []
        positionend = []
        # for i in range(timetable.shape[1]-1):
        #     if timetable[:, i].sum() == 0 and timetable[:, i+1].sum() >=1:
        #         positionstart.append(i)
        #     elif timetable[:, i].sum() >=1 and timetable[:, i+1].sum() ==0:
        #         positionend.append(i)
        for j in range(timetable.shape[0]):
            positionstart.append([])
            positionend.append([])
            for i in range(timetable.shape[1]-1):
                if timetable[j, i] == 0 and timetable[j, i+1] == 1:
                    positionstart[j].append(i)
                elif timetable[j, i]==1 and timetable[j, i+1]==0:
                    positionend[j].append(i)

        wavpath = get_line_context(wav_list, textgrid_num + 1) ##wav的列表
        wavid = wavpath.split('/')[-1]
        wavid = wavid.split('.wav')[0]
        y, fs = sf.read(wavpath)
        if not os.path.exists(output_dir+'/'+wavid):
            os.makedirs(output_dir+'/'+wavid)

        for j in range(len(positionstart)):
            for i in range(len(positionstart[j])):
                wavcut = y[positionstart[j][i]*160:positionend[j][i]*160]
                sf.write(output_dir+'/'+wavid+'/'+wavid+'-'+str(positionstart[j][i]).rjust(6,'0')+'-'+str(positionend[j][i]).rjust(6,'0')+'.wav',wavcut,fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_list",
                        type=str,
                        help="wav_list",
                        default="rawwav_list/test/wav.txt")
    parser.add_argument("--textgrid_list",
                        type=str,
                        help="textgrid_list",
                        default="rawwav_list/test/textgrid_test.txt")
    parser.add_argument("--output_dir",
                        type=str,
                        help="output_dir",
                        default="data_nospk")

    args = parser.parse_args()
    run(args)