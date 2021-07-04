#!/usr/bin/env python


import os
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_complex.functional as cF
from torch_complex.tensor import ComplexTensor
import sys
import linecache
import soundfile as sf
import librosa

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi

sys.path.append(os.path.dirname(sys.path[0]) + '/model')
# from feature_wj import STFT, iSTFT
from trans import STFT, iSTFT
sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from show import show_model, show_params

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = th.max(energy)
    low_thres = max_e * (10**(-50/10)) # to filter lower than 50dB 
    rms = th.mean(energy[energy >= low_thres])
    #rms = np.mean(energy)
    return rms

def snr(enh, noisy, eps=EPSILON):
    noise = enh - noisy
    return 10 * th.log10((rms(enh) + eps) / (rms(noise) + eps))

def trace(cplx_mat):
    """
    Return trace of a complex matrices
    """
    mat_size = cplx_mat.size()
    E = th.eye(mat_size[-1], dtype=th.bool).expand(*mat_size)
    return cplx_mat[E].view(*mat_size[:-1]).sum(-1)


def beamform(weight, spectrogram):
    """
    Do beamforming
    Args:
        weight: complex, N x C x F
        spectrogram: complex, N x C x F x T (output by STFT)
    Return:
        beam: complex, N x F x T
    """
    return (weight[..., None].conj() * spectrogram).sum(dim=1)


def estimate_covar(mask, spectrogram):
    """
    Covariance matrices (PSD) estimation
    Args:
        mask: TF-masks (real), N x F x T
        spectrogram: complex, N x C x F x T
    Return:
        covar: complex, N x F x C x C
    """
    # N x F x C x T
    spec = spectrogram.transpose(1, 2)
    # N x F x 1 x T
    mask = mask.unsqueeze(-2)
    # N x F x C x C
    nominator = cF.einsum("...it,...jt->...ij", [spec * mask, spec.conj()])
    # N x F x 1 x 1
    denominator = th.clamp(mask.sum(-1, keepdims=True), min=EPSILON)
    # N x F x C x C
    return nominator / denominator

class ChannelAttention(nn.Module):
    """
    Compute u for mvdr beamforming
    """

    def __init__(self, num_bins, att_dim):
        super(ChannelAttention, self).__init__()
        self.proj = nn.Linear(num_bins, att_dim)
        self.gvec = nn.Linear(att_dim, 1)

    def forward(self, Rs):
        """
        Args:
            Rs: complex, N x F x C x C
        Return:
            u: real, N x C
        """
        C = Rs.shape[-1]
        I = th.eye(C, device=Rs.device, dtype=th.bool)
        # diag is zero, N x F x C
        Rs = Rs.masked_fill(I, 0).sum(-1) / (C - 1)
        # N x C x A
        proj = self.proj(Rs.abs().transpose(1, 2))
        # N x C x 1
        gvec = self.gvec(th.tanh(proj))
        # N x C
        return F.softmax(gvec.squeeze(-1), -1)

class TorchRNNEncoder(nn.Module):
    """
    PyTorch's RNN encoder
    """
    def __init__(self,
                 input_size,
                 output_size,
                 input_project=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=1028,
                 rnn_dropout=0.2,
                 rnn_bidir=True,
                 non_linear="sigmoid"):
        super(TorchRNNEncoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        support_non_linear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh,
            "": None
        }
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if non_linear not in support_non_linear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_project:
            self.proj = nn.Linear(input_size, input_project)
        else:
            self.proj = None
        self.rnns = supported_rnn[RNN](
            input_size if input_project is None else input_project,
            rnn_hidden,
            rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=rnn_bidir)
        self.outp = nn.Linear(rnn_hidden if not rnn_bidir else rnn_hidden * 2,
                              output_size)
        self.non_linear = support_non_linear[non_linear]

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(self, inp, inp_len, max_len=None):
        """
        Args:
            inp (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        Return:
            out (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        """
        # self.rnns.flatten_parameters()
        if inp_len is not None:
            inp = pack_padded_sequence(inp,
                                       inp_len,
                                       batch_first=True,
                                       enforce_sorted=False)
        # extend dim when inference
        else:
            if inp.dim() not in [2, 3]:
                raise RuntimeError("TorchRNNEncoder expects 2/3D Tensor, " +
                                   f"got {inp.dim():d}")
            if inp.dim() != 3:
                inp = th.unsqueeze(inp, 0)
        if self.proj:
            inp = tf.relu(self.proj(inp))
        rnn_out, _ = self.rnns(inp)
        # using unpacked sequence
        # rnn_out: N x T x D
        if inp_len is not None:
            rnn_out, _ = pad_packed_sequence(rnn_out,
                                             batch_first=True,
                                             total_length=max_len)
        out = self.outp(rnn_out)
        # pass through non-linear
        if self.non_linear:
            out = self.non_linear(out)
        return out, inp_len

class MVDR(nn.Module):
    """
    BSS
    """

    def __init__(self, num_bins=257, att_dim=512, mask_norm=True):
        super(MVDR, self).__init__()
        # self.ref = ChannelAttention(num_bins, att_dim)
        self.mask_norm = mask_norm
        self.eps = EPSILON
        self.stft = STFT(frame_len=512, frame_hop=256)
        self.istft = iSTFT(frame_len=512, frame_hop=256)
        self.mask_net = TorchRNNEncoder(257*5, 257*2)
        show_model(self)
        show_params(self)

    def _derive_weight(self, Rs, Rn, u, eps=1e-5):
        """
        Compute mvdr beam weights
        Args:
            Rs, Rn: speech & noise covariance matrices, N x F x C x C
            u: reference selection vector, N x C
        Return:
            weight: N x F x C
        """
        C = Rn.shape[-1]
        I = th.eye(C, device=Rn.device, dtype=Rn.dtype)
        Rn = Rn + I * EPSILON
        # N x F x C x C
        Rn_inv = Rn.inverse()
        # N x F x C x C
        Rn_inv_Rs = cF.einsum("...ij,...jk->...ik", [Rn_inv, Rs])
        # N x F
        tr_Rn_inv_Rs = trace(Rn_inv_Rs) + EPSILON
        # N x F x C
        Rn_inv_Rs_u = cF.einsum("...fnc,...c->...fn", [Rn_inv_Rs, u])
        # N x F x C
        weight = Rn_inv_Rs_u / tr_Rn_inv_Rs[..., None]
        return weight

    def _process_mask(self, mask):
        """
        Process mask estimated by networks
        """
        if mask is None:
            return mask
        if self.mask_norm:
            max_abs = th.norm(mask, float("inf"), dim=1, keepdim=True)
            mask = mask / th.clamp(max_abs, EPSILON)
        mask = th.transpose(mask, 1, 2)
        return mask

    def _energy_accumulate(self, mask):
        energy = th.sum(mask**2)*0.999
        masksort = mask.reshape(-1)
        masksort = th.sort(masksort, descending=True)[0]
        accumulate = 0.0
        for i in range(masksort.shape[0]):
            accumulate = accumulate + masksort[i]**2
            if accumulate >= energy:
                threshold = masksort[i]
                break
        mask = mask * (mask >= threshold)
        return mask

    def forward(self, x, src, decode=False):
        """
        Args:
            mask_s: real TF-masks (speech), N x T x F
            x: noisy complex spectrogram, N x C x F x T
            mask_n: real TF-masks (noise), N x T x F
        Return:
            y: enhanced complex spectrogram N x T x F
        """
        inp_r, inp_i = self.stft(x)
        src_r, src_i = self.stft(src)
        mag, pha = th.sqrt(th.clamp(inp_r**2 + inp_i**2, EPSILON)), th.atan2(inp_i + EPSILON, inp_r + EPSILON)
        src_mag, src_pha = th.sqrt(th.clamp(src_r**2 + src_i**2, EPSILON)), th.atan2(src_i + EPSILON, src_r + EPSILON)
        ipd1 = th.cos(pha[:, 4]-pha[:, 0])
        ipd2 = th.cos(pha[:, 5]-pha[:, 1])
        ipd3 = th.cos(pha[:, 6]-pha[:, 2])
        ipd4 = th.cos(pha[:, 7]-pha[:, 3])

        magipd = th.cat([mag[:,0], ipd1, ipd2, ipd3, ipd4], 1)
        x_mask = self.mask_net(magipd.transpose(2,1), None)[0]
        mask_s1, mask_s2 = th.chunk(x_mask, 2, dim=-1)
        
        
        beam_out = th.stack([mask_s1.transpose(2,1), mask_s2.transpose(2,1)], 1)
        inf_mag1 = th.sqrt(th.clamp((inp_r[:,0]-src_r[:,0])**2 + (inp_i[:,0]-src_i[:,0])**2, EPSILON))
        src_out1 = src_mag[:,0] / th.clamp(th.sqrt(src_mag[:,0]**2 + inf_mag1**2), EPSILON)
        inf_mag2 = th.sqrt(th.clamp((inp_r[:,0]-src_r[:,1])**2 + (inp_i[:,0]-src_i[:,1])**2, EPSILON))
        src_out2 = src_mag[:,1] / th.clamp(th.sqrt(src_mag[:,1]**2 + inf_mag2**2), EPSILON)
        src_out = th.stack([src_out1, src_out2], 1)

    

        if decode:
            input_complex = ComplexTensor(inp_r, inp_i)
            mask_s1 = self._process_mask(mask_s1)
            mask_s2 = self._process_mask(mask_s2)
            mask_s1_masked = mask_s1# * (mask_s1 >= mask_s2)#mvdr
            mask_s2_masked= mask_s2 #* (mask_s1 < mask_s2)
            y1_mag = mag[:,0] * mask_s1_masked
            y2_mag = mag[:,0] * mask_s2_masked
            y1_pha = pha[:,0]
            y2_pha = pha[:,0]
            y1_real = y1_mag * th.cos(y1_pha)
            y1_imag = y1_mag * th.sin(y1_pha)
            y2_real = y2_mag * th.cos(y2_pha)
            y2_imag = y2_mag * th.sin(y2_pha)
            y1 = self.istft((y1_real, y1_imag))
            y2 = self.istft((y2_real, y2_imag))
            x = self.istft((inp_r[:,0], inp_i[:,0]))
            snr1 = snr(y1, x)
            snr2 = snr(y2, x)
            if snr1 > snr2:
                # mask = self._energy_accumulate(mask_s1)
                mask = mask_s1
                y = y1
            else:
                # mask = self._energy_accumulate(mask_s2)
                mask = mask_s2
                y = y2
            Rs = estimate_covar(mask, input_complex)
            Rn = estimate_covar(1.0-mask, input_complex)
            u = th.zeros([input_complex.shape[0],input_complex.shape[1]], dtype=th.float32, device = input_complex.device)
            u[:, 0] = 1.0
            weight = self._derive_weight(Rs, Rn, u, eps=EPSILON)
            # N x C x F
            weight = weight.transpose(1, 2)
            # N x F x C
            beam = beamform(weight, input_complex)
            beam = self.istft((beam.real, beam.imag))
            return beam
            
        return beam_out, src_out

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params



def calloss_mse(output, source):
    loss_total = 0.0

    for i in range(output.shape[0]):
        loss1 = F.mse_loss(output[i,0], source[i,0], reduction='sum')
        loss2 = F.mse_loss(output[i,1], source[i,1], reduction='sum')
        loss3 = F.mse_loss(output[i,0], source[i,1], reduction='sum')
        loss4 = F.mse_loss(output[i,1], source[i,0], reduction='sum')
        loss = th.min((loss1 + loss2) / 2, (loss3 + loss4) / 2)
        loss = loss / output.shape[3]
        loss_total = loss_total + loss
    loss_total = loss_total / output.shape[0]
    return loss_total



def main():
    nnet = MVDR()

    x = th.randn([8,8,64000]).contiguous()
    y=nnet(x.contiguous(), x[:,:2].contiguous())

    # fn = r'/home/work_nfs3/yhfu/workspace/dccrn_upe_slt/data_2mic_hw/tt/se_nodrv.scp'
    # fc = r'/home/work_nfs3/yhfu/workspace/dccrn_upe_slt/data_2mic_hw/tt/se_nodrv_label.scp'
    # for i in range(2000):
    #     npath = get_line_context(fn, i + 1)
    #     cpath = get_line_context(fc, i + 1)
    #     key, noisy_path = npath.split(' ')
    #     key, clean_path = cpath.split(' ')
    #     y,fs=sf.read(noisy_path)
    #     clean,fs=sf.read(clean_path)

    #     # y = th.from_numpy(y)
    #     # clean = th.from_numpy(clean)
    #     # y = y.unsqueeze(0).transpose(2, 1).float()
    #     # clean = clean.unsqueeze(0).transpose(2, 1).float()
    #     # enh = nnet(y, clean)
    #     # print(enh.shape)
    #     # sf.write('/home/work_nfs3/yhfu/workspace/dccrn_upe_slt/exphw/signal_mvdr/'+key+'_1.wav', enh.squeeze().numpy(), fs)
    #     y = librosa.stft(y[:,0], 512, 256)
    #     clean = librosa.stft(clean[:,0], 512, 256)
    #     mag_y = np.sqrt(y.real**2 + y.imag**2 + 1e-7)
    #     mag_clean = np.sqrt(clean.real**2 + clean.imag**2)
    #     mask = mag_clean / mag_y
    #     np.save('/home/work_nfs3/yhfu/workspace/dccrn_upe_slt/exphw/signal_mvdr/mask/'+key+'.npy',mask)




if __name__ == '__main__':
    main()
