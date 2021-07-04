#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import numpy as np
import torch as th

from pathlib import Path
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.opts import DecodingParser
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader
"""
Nbest format:
Number: n
key1
score-1 num-tok-in-hyp-1 hyp-1
...
score-n num-tok-in-hyp-n hyp-n
...
keyM
score-1 num-tok-in-hyp-1 hyp-1
...
score-n num-tok-in-hyp-n hyp-n
"""

logger = get_logger(__name__)

beam_search_params = [
	"beam_size", "nbest", "max_len", "min_len", "max_len_ratio",
	"min_len_ratio", "len_norm", "lm_weight", "temperature", "len_penalty",
	"cov_penalty", "eos_threshold", "cov_threshold", 
]
def drc(input, fs, gain=20, full_gain_pos=0.01, limit=0.97):

	if len(input.shape) > 1:
		input = input[:,0]
	if limit > 0.99:
		raise RuntimeError("Please set 'limit' to smaller than 0.99")
	if full_gain_pos == 0:
		full_gain_pos = 1e-10

	framesize = fs//1000 # 1ms
	CompressThreshold = limit*0.9
	nframes = input.shape[0] // framesize
	SmoothedPeak = 0
	PeakSmoothFactor = 0.99

	prevGain = 1
	result = np.zeros_like(input)    
	buffer = input[:framesize]
	for idx in range(nframes-2):
		frame = input[idx*framesize:(idx+1)*framesize]
		peak = np.max(np.abs(frame)) 
		#level = 0 
		if peak < SmoothedPeak:
			level = SmoothedPeak
			SmoothedPeak = SmoothedPeak*PeakSmoothFactor + (1 - PeakSmoothFactor)*peak 
		else:
			level = peak 
			SmoothedPeak = peak 
		
		target_level = level * gain 
		
		if target_level > CompressThreshold:
			newlevel = CompressThreshold \
							+(target_level - CompressThreshold) \
							/(gain -CompressThreshold) \
							*(limit - CompressThreshold)

			newlevel = np.min([newlevel, limit*0.9999])
			newGain = newlevel/level
		elif target_level > full_gain_pos:
			newGain = gain 
		else:
			newGain = 1+(gain-1)*target_level/full_gain_pos 

		assert(level * newGain <=limit)

		gainInterp = prevGain + (newGain - prevGain) * np.arange(framesize)/framesize
		
		result[idx*framesize:(idx+1)*framesize] = buffer*gainInterp
		buffer = frame
		prevGain = newGain

	return result


class FasterDecoder(NnetEvaluator):
	"""
	Decoder wrapper
	"""

	def __init__(self,
				 cpt_dir: str,
				 cpt_tag: str = "best",
				 function: str = "beam_search",
				 device_id: int = -1) -> None:
		super(FasterDecoder, self).__init__(cpt_dir,
											task="asr",
											cpt_tag=cpt_tag,
											device_id=device_id)
		if not hasattr(self.nnet, function):
			raise RuntimeError(
				f"AM doesn't have the decoding function: {function}")
		self.decode = getattr(self.nnet, function)
		self.function = function
		logger.info(f"Load checkpoint from {cpt_dir}, epoch: " +
					f"{self.epoch}, tag: {cpt_tag}")
		logger.info(f"Using decoding function: {function}")

	def run(self, src, **kwargs):
		src = th.from_numpy(src).to(self.device)
		if self.function == "greedy_search":
			return self.decode(src)
		else:
			return self.decode(src, **kwargs)


def run(args):
	print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
	split_by_time = True
	decoder = FasterDecoder(args.am,
							cpt_tag=args.am_tag,
							function=args.function,
							device_id=args.device_id)
	if decoder.accept_raw:
		src_reader = AudioReader(wav_scp=args.feats_or_wav_scp,
				#args.feats_or_wav_scp,
				#args.feats_or_wav_scp,
								 sr=16000,#args.sr,
								 channel=args.channel)
	else:
		src_reader = ScriptReader(args.feats_or_wav_scp)

	if args.lm:
		if Path(args.lm).is_file():
			from aps.asr.lm.ngram import NgramLM
			lm = NgramLM(args.lm, args.dict)
			logger.info(
				f"Load ngram LM from {args.lm}, weight = {args.lm_weight}")
		else:
			lm = NnetEvaluator(args.lm,
							   device_id=args.device_id,
							   cpt_tag=args.lm_tag)
			logger.info(f"Load RNN LM from {args.lm}: epoch {lm.epoch}, " +
						f"weight = {args.lm_weight}")
			lm = lm.nnet
	else:
		lm = None

	processor = TextPostProcessor(args.dict,
								  space=args.space,
								  show_unk=args.show_unk,
								  spm=args.spm)
	stdout_top1, top1 = io_wrapper(args.best, "w")

	if split_by_time:
		stdout_time, time1 = io_wrapper(args.best + '_time_format', 'w')
	topn = None
	if args.dump_nbest:
		stdout_topn, topn = io_wrapper(args.dump_nbest, "w")
		if args.function == "greedy_search":
			nbest = min(args.beam_size, args.nbest)
		else:
			nbest = 1
		topn.write(f"{nbest}\n")
	ali_dir = args.dump_align
	if ali_dir:
		Path(ali_dir).mkdir(exist_ok=True, parents=True)
		logger.info(f"Dump alignments to dir: {ali_dir}")
	N = 0
	timer = SimpleTimer()
	dec_args = dict(
		filter(lambda x: x[0] in beam_search_params,
			   vars(args).items()))
	dec_args["lm"] = lm
	#time_length_file = open('/home/environment/yhfu/aishell4_release/test_segment', 'r')
	#lines = time_length_file.readlines()
	for key, src in src_reader:
		src = drc(src, 16000, 5, 0.01, 0.97)
		#print('src', src.shape, flush=True)
		#utt_id, wavid, s_time, e_time = lines[cnt_time_length].strip().split(' ')
		if split_by_time:
			try:
				if len(key.strip().split('-')) == 3:
					utt_id, s_time, e_time = key.strip().split('-')
				else:
					utt_id, s_time, e_time, _ = key.strip().split('-')
				s_time = int(s_time) * 0.01
				e_time = int(e_time) * 0.01
				t_wav = len(src)/16000
			except ValueError:
				utt_id = key.strip()
				s_time = 0
				t_wav = len(src)/16000
				e_time = s_time + t_wav
		logger.info(f"Decoding utterance {key}...")
		
		nbest_hypos = decoder.run(src, **dec_args)
		nbest = [f"{key}\n"]
		for idx, hyp in enumerate(nbest_hypos):
			# remove SOS/EOS
			token = hyp["trans"][1:-1]
			trans = processor.run(token)
			score = hyp["score"]
			nbest.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
			if idx == 0:
				if split_by_time:
					######format
					words = []
					for tran in trans:
						if str(tran) != ' ':
							#print('trans', tran, ' ', len(trans), type(trans), flush=True)
							words.append(str(tran))

					t_word = float(t_wav / len(words))
					#time1.write(key + '\t')
					t_start = float(s_time)
					for word in words:
						time1.write(f"{utt_id} 0 {t_start:.2f} {t_word:.2f} {word}\n")
						t_start += t_word
					#######
				top1.write(f"{key}\t{trans}\n")
			if ali_dir:
				if hyp["align"] is None:
					raise RuntimeError(
						"Can not dump alignment out as it's None")
				np.save(f"{ali_dir}/{key}-nbest{idx+1}", hyp["align"].numpy())
		if topn:
			topn.write("".join(nbest))
		if not (N + 1) % 10:
			top1.flush()
			if split_by_time:
				time1.flush()
			if topn:
				topn.flush()
		N += 1
	if not stdout_top1:
		top1.close()
		if split_by_time:
			time1.close()
	if topn and not stdout_topn:
		topn.close()
	cost = timer.elapsed()
	logger.info(
		f"Decode {len(src_reader)} utterance done, time cost = {cost:.2f}m")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=
		"Command to do end-to-end decoding using beam search algothrim",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		parents=[DecodingParser.parser])
	parser.add_argument("--function",
						type=str,
						choices=["beam_search", "greedy_search"],
						default="beam_search",
						help="Name of the decoding function")
	args = parser.parse_args()
	run(args)
