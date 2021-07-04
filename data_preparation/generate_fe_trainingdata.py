#!/usr/bin/env python

import io
import os
import subprocess
import linecache
import numpy as np
import soundfile as sf
import scipy.signal as ss
import random
import time
import librosa
import argparse

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def sfread(fname):
    y, fs = sf.read(fname)
    if fs != 16000:
        y = librosa.resample(y, fs, 16000)
    return y

def cutwav(wav, minlen, maxlen):
    if wav.shape[0] < 16000*maxlen:
        return wav
    else:
        duration = int(random.uniform(minlen,maxlen)*16000)
        start = random.randint(0, wav.shape[0]-duration)
        wav = wav[start:start+duration]
        return wav

def mixwav(fname1, fname2, fnoisename, frir, fisotropic, ratio, snr, sir, isosnr):
    samps1= cutwav(sfread(fname1), 3, 5)
    samps2 = cutwav(sfread(fname2), 3, 5)
    noise = cutwav(sfread(fnoisename), 5, 10)
    rir = sfread(frir)
    isotropic = sfread(fisotropic)
    if len(samps1.shape) > 1:
        samps1 = samps1[:,0]
    if len(samps2.shape) > 1:
        samps2 = samps2[:,0]
    if len(noise.shape) > 1:
        noise = noise[:,0]
    
    ratio = float(ratio)
    snr = float(snr)
    sir = float(sir)
    isosnr = float(isosnr)

    overlaplength = int(ratio*(samps1.shape[0] + samps2.shape[0])/2)

    padlength1 = samps2.shape[0] - overlaplength
    padlength2 = samps1.shape[0] - overlaplength

    if padlength1 > 0 and padlength2 > 0:
        samps1 = np.pad(samps1,(0,padlength1),'constant', constant_values=(0,0))
        samps2 = np.pad(samps2,(padlength2,0),'constant', constant_values=(0,0))
    samps, samps1, samps2 = add_noise(samps1, samps2, rir, sir)
    samps, _, _ = add_noise(samps, noise, rir, snr)
    samps, _, _ = add_noise(samps, isotropic, rir, isosnr)
    return samps, samps1, samps2



def mix_snr(clean, noise, snr):
    t = np.random.normal(loc = 0.9, scale = 0.1)
    if t < 0:
        t = 1e-1
    elif t > 1:
        t = 1
    scale = t

    clean_snr = snr
    noise_snr = -snr

    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    for i in range(clean.shape[1]):
        clean[:, i]  = activelev(clean[:, i]) * clean_weight
        noise[:, i]  = activelev(noise[:, i]) * noise_weight
    noisy = clean + noise

    max_amp = np.zeros(clean.shape[1])
    for i in range(clean.shape[1]):
        max_amp[i] = np.max(np.abs([clean[:,i], noise[:,i], noisy[:,i]]))
        if max_amp[i] == 0:
            max_amp[i] = 1
        max_amp[i] = 1 / max_amp[i] * scale

    for i in range(noisy.shape[1]):
        noisy[:, i]= noisy[:, i] * max_amp[i]
        clean[:, i]= clean[:, i] * max_amp[i]
        noise[:, i]= noise[:, i] * max_amp[i]
    
    return noisy, clean, noise

def add_reverb(cln_wav, rir_wav):
    """
    Args:
        :@param cln_wav: the clean wav
        :@param rir_wav: the rir wav
    Return:
        :@param wav_tgt: the reverberant signal
    """
    rir_wav = np.array(rir_wav)
    wav_tgt = np.zeros([cln_wav.shape[0]+7999, rir_wav.shape[1]])
    for i in range(rir_wav.shape[1]):
        wav_tgt[:, i] = ss.oaconvolve(cln_wav, rir_wav[:,i]/np.max(np.abs(rir_wav[:,i])))
    return wav_tgt

def activelev(data):
    # max_val = np.max(np.abs(data))
    max_val = np.std(data)
    if max_val == 0:
        return data
    else:
        return data / max_val

def add_noise(clean, noise, rir, snr):
    random.seed(time.clock())
    if len(noise.shape) == 1 and len(clean.shape) > 1:
        noise = add_reverb(noise, rir[:, 16:24])
        noise = noise[:-7999]
        snr = snr / 2
        flag = 'addnoise'
    elif len(noise.shape) == 1 and len(clean.shape) == 1:
        clean = add_reverb(clean, rir[:, 0:8])
        noise = add_reverb(noise, rir[:, 8:16])
        clean = clean[:-7999]
        noise = noise[:-7999]
        flag = 'twospk'
    else:
        snr = snr / 2
        flag = 'iso'

    clean_length = clean.shape[0]
    noise_length = noise.shape[0]

    if clean_length > noise_length:
        padlength = clean_length - noise_length
        padfront = random.randint(0, padlength)
        padend =  padlength - padfront
        noise = np.pad(noise, ((padfront, padend), (0, 0)),'constant', constant_values=(0,0))
        noise_selected = noise
        clean_selected = clean
        
    elif clean_length < noise_length and flag == 'twospk': 
        padlength = noise_length - clean_length
        padfront = random.randint(0, padlength)
        padend =  padlength - padfront
        clean = np.pad(clean, ((padfront, padend), (0, 0)),'constant', constant_values=(0,0))
        noise_selected = noise
        clean_selected = clean
        
    elif clean_length < noise_length and (flag == 'addnoise' or flag == 'iso'): 
        start = random.randint(0, noise_length - clean_length)
        noise = noise[start:start+clean_length]
        noise_selected = noise
        clean_selected = clean

    
    else:
        noise_selected = noise
        clean_selected = clean

    noisy, clean, noise = mix_snr(clean_selected, noise_selected, snr)
    return noisy, clean, noise


def run(args):
    wavlist1 = args.spk1_list
    wavlist2 = args.spk2_list
    noiselist = args.noise_list
    rirlist = args.rir_list
    isolist = args.isotropic_list
    datamode = args.mode
    output_dir = args.output_dir

    utt2dur = open(output_dir+'/'+datamode+'/utt2dur', 'w')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/'+datamode+'/mix'):
        os.makedirs(output_dir+'/'+datamode+'/mix')
        os.makedirs(output_dir+'/'+datamode+'/spk1')
        os.makedirs(output_dir+'/'+datamode+'/spk2')


    for i in range(args.wavnum):

        random.seed(time.clock())
        wav1idx = random.randint(0, len(open(wavlist1,'r').readlines())-1)
        wav2idx = random.randint(0, len(open(wavlist2,'r').readlines())-1)
        noiseidx = random.randint(0, len(open(noiselist,'r').readlines())-1)
        riridx = random.randint(0, len(open(rirlist,'r').readlines())-1)
        isotropicidx = random.randint(0, len(open(isolist,'r').readlines())-1)
        wav1_path = get_line_context(wavlist1, wav1idx+1)
        wav2_path = get_line_context(wavlist2, wav2idx+1)
        noise_path = get_line_context(noiselist, noiseidx+1)
        rir_path = get_line_context(rirlist, riridx+1)
        isotropic_path = get_line_context(isolist, isotropicidx+1)
        random.seed(time.clock())
        snr = random.uniform(5, 20)
        sir = random.uniform(-5, 5)
        isosnr = random.uniform(15,25)
        scenario = random.randint(0, 2)
        if scenario == 0:
            ratio = random.uniform(0, 0.2)
        elif scenario == 1:
            ratio = random.uniform(0.2, 0.8)
        elif scenario == 2:
            ratio = 0.0
        outname = str(i+1).rjust(5,'0')+'.wav'
        out, spk1, spk2 = mixwav(wav1_path, wav2_path, noise_path, rir_path, isotropic_path, ratio, snr, sir, isosnr)
        sf.write(output_dir+'/'+datamode+'/mix/'+outname, out, 16000)
        sf.write(output_dir+'/'+datamode+'/spk1/'+outname, spk1[:,0], 16000)
        sf.write(output_dir+'/'+datamode+'/spk2/'+outname, spk2[:,0], 16000)
        utt2dur.write(outname.split('.wav')[0]+' '+str(out.shape[0]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spk1_list",
                        type=str,
                        help="spk1_list",
                        default="rawwav_list/train/librispeech_tr_spk1.txt")
    parser.add_argument("--spk2_list",
                        type=str,
                        help="spk2_list",
                        default="rawwav_list/train/librispeech_tr_spk2.txt")
    parser.add_argument("--noise_list",
                        type=str,
                        help="noise_list",
                        default="rawwav_list/train/noise_tr.txt")
    parser.add_argument("--rir_list",
                        type=str,
                        help="rir_list",
                        default="rawwav_list/train/rir_2-8s_1-5m_aishell4_tr.txt")
    parser.add_argument("--isotropic_list",
                        type=str,
                        help="isotropic_list",
                        default="rawwav_list/train/iso_tr.txt")
    parser.add_argument("--mode",
                        type=str,
                        help="train or dev",
                        default="train")
    parser.add_argument("--output_dir",
                        type=str,
                        help="output_dir for data",
                        default="data_frontend")
    parser.add_argument("--wavnum",
                        type=int,
                        help="total number of simulated wavs",
                        default=100)
    args = parser.parse_args()
    run(args)