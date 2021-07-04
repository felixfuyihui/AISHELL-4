#!/usr/bin/env python -u
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys
import math

import torch
import torch.nn as nn

import numpy as np
import torch.optim as optim
import torch.nn.parallel.data_parallel as data_parallel
import torch.nn.functional as F
from torch import autograd
import librosa
import soundfile as sf
from subprocess import call

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'utils'))
import misc.logging as logger
from base.dataset import TimeDomainDateset
from base.data_reader import DataReader
from misc.common import pp, str_to_bool
from model.misc import save_checkpoint, reload_model, reload_for_eval
from model.misc import learning_rate_decaying, get_learning_rate
from model.estimask import MVDR, calloss_mse
from sigproc.sigproc import wavwrite, wavread

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(magnitude, name):
        plt.imshow(magnitude)
        plt.savefig(name)

SAMPLE_RATE = 16000
CLIP_SIZE = 4   #segmental length is 4s 12s
SAMPLE_LENGTH = SAMPLE_RATE * CLIP_SIZE

# torch.cuda.set_device("5,6")
def train(model, device, writer):
    fid = open(FLAGS.model_dir+'/traininglog3'+'.txt','a')
    print('preparing data...')
    fid.writelines('preparing data...\n')
    fid.flush()
    mix_scp = FLAGS.mix_scp_tr
    s1_scp = FLAGS.s1_scp_tr
    s2_scp = FLAGS.s2_scp_tr
    utt2dur = FLAGS.utt2dur_tr
    dataset = TimeDomainDateset(mix_scp, s1_scp, s2_scp, utt2dur, SAMPLE_RATE, CLIP_SIZE)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=True)

    print_freq = 10
    batch_num = len(dataloader)
    print(batch_num)
    start_epoch = 0
 
    optimizer1 = optim.Adam(model.get_params(FLAGS.weight_decay), lr=FLAGS.lr)

    # reload previous model
    start_epoch, start_step = reload_model(model, optimizer1, FLAGS.model_dir, FLAGS.use_cuda)
    step = start_step
    lr = get_learning_rate(optimizer1)

    print('=> RERUN', end=' ')
    fid.writelines('=> RERUN\n')
    fid.flush()
    val_loss = validation(model, -1, lr, device)
    best = val_loss
    # best = 10000.0

    

    for epoch in range(start_epoch, FLAGS.epochs):
        # Set random seed
        torch.manual_seed(FLAGS.seed + epoch)
        if FLAGS.use_cuda:
            torch.cuda.manual_seed(FLAGS.seed + epoch)
        model.train()
        loss_total1 = 0.0
        loss_print1 = 0.0
        start_time = datetime.datetime.now()
        lr = get_learning_rate(optimizer1)
        for idx, data in enumerate(dataloader):
            mix = data['mix'].to(device)
            src = data['src'].to(device)

            mix = mix.to(device)
            src = src.to(device)

            optimizer1.zero_grad()
            mix = mix.float()
            src = src.float()
           
            output_mag, src_mag = data_parallel(model, (mix, src, False))
            loss1 = calloss_mse(output_mag, src_mag)
            # with autograd.detect_anomaly():
            loss1.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer1.step()

            step = step + 1
            loss_total1 = loss_total1 + loss1.data.cpu()
            loss_print1 = loss_print1 + loss1.data.cpu()
            

            if (idx) % print_freq == 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                speed_avg = elapsed / (idx+1)
                loss_print_avg1 = loss_print1 / print_freq
                print('Epoch {:2d}/{:2d} | batches {:4d}/{:4d} | lr {:1.4e} | '
                      '{:2.3f} s/batch | LOSS {:2.8f}'.format(
                          epoch + 1, FLAGS.epochs, idx, batch_num, lr,
                          speed_avg, loss_print_avg1))
                fid.writelines('Epoch {:2d}/{:2d} | batches {:4d}/{:4d} | lr {:1.4e} | '
                      '{:2.3f} s/batch | LOSS {:2.8f}\n'.format(
                          epoch + 1, FLAGS.epochs, idx, batch_num, lr,
                          speed_avg, loss_print_avg1))
                fid.flush()
                writer.add_scalar('Loss/Train', loss_print_avg1, step)
                sys.stdout.flush()
                loss_print1 = 0.0
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        speed_avg = elapsed / batch_num
        loss_total_avg1 = loss_total1 / batch_num
        print('TRAINING AVG.LOSS | epoch {:3d}/{:3d} | step {:7d} | lr  {:1.4e} | '
              '{:2.3f} s/batch | time {:3.2f} mins | LOSS {:2.8f}'.format(
                  epoch + 1, FLAGS.epochs, step, lr, speed_avg, elapsed / 60.0,
                  loss_total_avg1.item()))
        fid.writelines('TRAINING AVG.LOSS | epoch {:3d}/{:3d} | step {:7d} | lr  {:1.4e} | '
              '{:2.3f} s/batch | time {:3.2f} mins | LOSS {:2.8f}\n'.format(
                  epoch + 1, FLAGS.epochs, step, lr, speed_avg, elapsed / 60.0,
                  loss_total_avg1.item()))
        fid.flush()

        # Do cross validation
        val_loss  = validation(model, epoch, lr, device)
        writer.add_scalar('Loss/Valid', val_loss, step)
        fid.writelines('Loss/Valid LOSS is {:2.8f})\n'.format(
                val_loss))
        fid.flush()

        if val_loss > best:
            learning_rate_decaying(optimizer1, 0.5)
            print('(Nnet rejected, the best LOSS is {:2.8f})'.format(
                best))
            fid.writelines('(Nnet rejected, the best LOSS is {:2.8f})\n'.format(
                best))
            fid.flush()
            save_checkpoint(model, optimizer1, epoch + 1, step, FLAGS.model_dir)
        else:
            print('(Nnet accepted)')
            fid.writelines('(Nnet accepted)\n')
            fid.flush()
            save_checkpoint(model, optimizer1, epoch + 1, step, FLAGS.model_dir)
        best = val_loss
        # Decaying learning rate

        sys.stdout.flush()
        start_time = datetime.datetime.now()


def validation(model,epoch, lr, device):
    mix_scp = FLAGS.mix_scp_dev
    s1_scp = FLAGS.s1_scp_dev
    s2_scp = FLAGS.s2_scp_dev
    utt2dur = FLAGS.utt2dur_dev
    dataset = TimeDomainDateset(mix_scp, s1_scp, s2_scp, utt2dur, SAMPLE_RATE, CLIP_SIZE)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=True)
    model.eval()
    loss_total = 0.0
    batch_num = len(dataloader)
    start_time = datetime.datetime.now()
    # start_data = datetime.datetime.now()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            mix = data['mix'].to(device)
            src = data['src'].to(device)


            mix = mix.float()
            src = src.float()
            
            output_mag, src_mag = data_parallel(model, (mix, src, False))
            loss = calloss_mse(output_mag, src_mag)
            loss_total = loss_total + loss.data.cpu()

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        speed_avg = elapsed / batch_num
        loss_total_avg = loss_total / batch_num
    print('CROSSVAL AVG.LOSS | epoch {:3d}/{:3d} | lr {:1.4e} | '
          '{:2.3f} s/batch | time {:2.1f} mins | LOSS {:2.8f}'.format(
              epoch + 1, FLAGS.epochs, lr, speed_avg, elapsed / 60.0,
              loss_total_avg),
          end=' ')
    return loss_total_avg

def evaluate(model, device):
    # Turn on evaluation mode which disables dropout.
    # split into small trunk then combine
    model.eval()
    mix_scp = FLAGS.wav_scp_test
    dataset = DataReader(mix_scp)
    filename = mix_scp.split('.scp')[0]

    total_num = len(dataset)
    print(total_num)
    # filename = 'wav_spk_e15'
    save_path = os.path.join(FLAGS.model_dir, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('=> Decoding ...')
    with torch.no_grad():
        for idx, data in enumerate(dataset.read()):
            key = data['key']
            print(key)
            mix = data['mix'].to(device).float()
            max_norm = data['max_norm']
            output = model(mix, mix, decode=True)
            output = output / torch.max(torch.abs(output)) * max_norm
            output = output.squeeze().cpu().numpy()
            wavwrite(output, SAMPLE_RATE, save_path+'/'+key + '.wav')
           

def build_model(decode):
    model = MVDR()
    return model


def main():
    device = torch.device('cuda' if FLAGS.use_cuda else 'cpu')
    model = build_model(FLAGS.decode)
    model.to(device)

    if FLAGS.logdir is None:
        writer = SummaryWriter(FLAGS.model_dir + '/tensorboard')
    else:
        writer = SummaryWriter(FLAGS.logdir)

    # Training
    if not FLAGS.decode:
        train(model, device, writer)
    # Evaluating
    else:
        reload_for_eval(model, FLAGS.model_dir, FLAGS.use_cuda)
        evaluate(model, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Mini-batch size')
    parser.add_argument(
        '--learning-rate',
        dest='lr',
        type=float,
        default=1e-3,
        help='Inital learning rate')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Max training epochs')
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay for optimizer (L2 penalty)')
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        type=str,
        required=True,
        help='Model directory')
    parser.add_argument(
        '--logDir',
        dest='logdir',
        type=str,
        default=None,
        help='Log directory (for tensorboard)')
    parser.add_argument(
        '--use-cuda',
        dest='use_cuda',
        type=str_to_bool,
        default=True,
        help='Enable CUDA training')
    parser.add_argument(
        '--decode',
        type=str_to_bool,
        default=False,
        help='Flag indicating decoding or training')
    parser.add_argument(
        '--mix-scp-tr',
        dest='mix_scp_tr',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--s1-scp-tr',
        dest='s1_scp_tr',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--s2-scp-tr',
        dest='s2_scp_tr',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--utt2dur-tr',
        dest='utt2dur_tr',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--mix-scp-dev',
        dest='mix_scp_dev',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--s1-scp-dev',
        dest='s1_scp_dev',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--s2-scp-dev',
        dest='s2_scp_dev',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--utt2dur-dev',
        dest='utt2dur_dev',
        type=str,
        default="mix.scp",
        help='')
    parser.add_argument(
        '--wav-scp-test',
        dest='wav_scp_test',
        type=str,
        default="mix.scp",
        help='')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    print('*** Parsed arguments ***')
    pp.pprint(FLAGS.__dict__)
    os.makedirs(FLAGS.model_dir, exist_ok=True)
    # Set the random seed manually for reproducibility.
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    logger.set_verbosity(logger.INFO)
    main()
