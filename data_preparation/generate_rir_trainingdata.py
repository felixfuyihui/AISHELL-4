#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for computing the RIR between several sources and receivers in GPU.
@author: yhfu@npu-aslp.org
pyrirgen: https://github.com/phecda-xu/RIR-Generator
"""

import numpy as np
import soundfile as sf
import math
import pyrirgen
import argparse
import os

def run(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    c = 340
    fs=16000 # Sampling frequency [Hz]
    num_room = 2500
    utt_per_room = 5
    room_x = 10
    room_y = 10
    room_z = 3
    nb_src = 3  # Number of sources
    nb_rcv = 8 # Number of receivers

    for i in range(num_room):
        x = np.random.uniform(3, room_x)
        y = np.random.uniform(3, room_y)
        z = room_z
        for j in range(utt_per_room):
            mic_distance = 0.05
            room_sz = [x, y, z]  # Size of the room [m]
            pos_src1 = [np.random.uniform(0, x),np.random.uniform(0, y),np.random.uniform(1.2, 1.9)]
            pos_src2 = [np.random.uniform(0, x),np.random.uniform(0, y),np.random.uniform(1.2, 1.9)]
            pos_src3 = [np.random.uniform(0, x),np.random.uniform(0, y),np.random.uniform(0, z)] # Positions of the sources ([m]
            
            mic_middle_point = [np.random.uniform(x/2-1.0, x/2+1.0),
                                np.random.uniform(y/2-1.0, y/2+1.0),
                                np.random.uniform(0.6, 1.2)]

            
            baseangle = 0.125*np.pi
            pos_rcv = [[mic_middle_point[0]+mic_distance*np.cos(16*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*16), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(14*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*14), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(12*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*12), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(10*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*10), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(8*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*8), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(6*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*6), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(4*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*4), mic_middle_point[2]],
                       [mic_middle_point[0]+mic_distance*np.cos(2*baseangle), mic_middle_point[1]+mic_distance*np.sin(baseangle*2), mic_middle_point[2]],]

                       
            mic_pattern = "omnidirectional" # Receiver polar pattern
            T60 = np.random.uniform(0.2, 0.8)    # Time for the RIR to reach 60dB of attenuation [s]

            RIRs1 = pyrirgen.generateRir(room_sz, pos_src1, pos_rcv, soundVelocity=c, fs=fs, reverbTime = T60, nSamples = 8000, 
                                        micType = mic_pattern, nOrder=-1, nDim=3, isHighPassFilter=True) #source * mic * time
            RIRs2 = pyrirgen.generateRir(room_sz, pos_src2, pos_rcv, soundVelocity=c, fs=fs, reverbTime = T60, nSamples = 8000, 
                                        micType = mic_pattern, nOrder=-1, nDim=3, isHighPassFilter=True) #source * mic * time
            RIRs3 = pyrirgen.generateRir(room_sz, pos_src3, pos_rcv, soundVelocity=c, fs=fs, reverbTime = T60, nSamples = 8000, 
                                        micType = mic_pattern, nOrder=-1, nDim=3, isHighPassFilter=True) #source * mic * time
            RIRs1=np.array(RIRs1)
            RIRs2=np.array(RIRs2)
            RIRs3=np.array(RIRs3)
            
            out = np.zeros([24, 8000])
            out[0:8]= RIRs1
            out[8:16]= RIRs2
            out[16:24]= RIRs3
            out = out.transpose(1,0)
            pos_src = np.array(pos_src1)
            pos_src2 = np.array(pos_src2)
            pos_src3 = np.array(pos_src3)
            pos_src = np.stack((pos_src1, pos_src2, pos_src3), axis=0)

            mic_middle_point = np.array(mic_middle_point)
            distance = np.zeros(nb_src)
            angle = np.zeros(nb_src)
            
            for k in range(nb_src):
                # distance[k] = np.linalg.norm(pos_src[k]-mic_middle_point)
                distance[k] = np.sqrt((pos_src[k][0]-mic_middle_point[0])**2 + (pos_src[k][1]-mic_middle_point[1])**2)
                angle[k] = np.arctan2((pos_src[k][1]-mic_middle_point[1]), (pos_src[k][0]-mic_middle_point[0]))
                angle[k] = math.degrees(angle[k])
                if angle[k]<0:
                	angle[k] = 360-np.abs(angle[k])
            matrix_1m = np.array([0.5, 0.5, 0.5])
            matrix_5m = np.array([5.0, 5.0, 5.0])
            if (distance>matrix_1m).all() and (distance<matrix_5m).all() and abs(angle[1] - angle[0])>20.0 and abs(angle[2] - angle[0])>20.0 and abs(angle[2] - angle[1])>20.0:
                wav_name = args.output_dir+'/' + \
                           ('%.2f' % x) + '_' + ('%.2f' % y) + '_' + ('%.2f' % z) + '_' + \
                           ('%.2f' % distance[0]) + '_' + ('%.2f' % distance[1]) + '_' + \
                           ('%.4f' % angle[0]) + '_' + ('%.4f' % angle[1])+ '_' + ('%.4f' % T60) + '.wav'
                sf.write(wav_name,out,16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        type=str,
                        help="output_dir",
                        default="rir")
    args = parser.parse_args()
    run(args)
