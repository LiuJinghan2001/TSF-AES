'''
This part is used to train the semi-supervised speaker model with S2 and S3 of TSF-AES and evaluate the performances
'''

import sys, tqdm, soundfile, time
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import gc
from itertools import cycle

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, putoff, epoch, loader1, loader2, loader3):
		self.train()

		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		tstart = time.time()
		index, top1, loss, loss1, loss2, quality, quantity, minprob, minindex, indexulb, num = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		num = 1
		if putoff < 1:   # Stage3
			for (idx, data, labels), (x_ulb_idx,  x_ulb_w,x_ulb_s, y_ulb) in zip(cycle(loader1), loader2):
			#for (idx, data, labels), (x_ulb_idx, x_ulb_w, x_ulb_s, y_ulb) in zip(loader1, loader2):
				self.zero_grad()
				x_ulb_idx = x_ulb_idx.cuda()
				labels = torch.LongTensor(labels).cuda()
				y_ulb = torch.LongTensor(y_ulb).cuda()
				speaker_embedding, logits_lb = self.speaker_encoder.forward(data.cuda(), aug=True)
				speaker_embedding1, logits_ulb_s = self.speaker_encoder.forward(x_ulb_s.cuda(), aug=True)
				speaker_embedding2, logits_ulb_w = self.speaker_encoder.forward(x_ulb_w.cuda(), aug=True)
				nloss1, prec = self.speaker_loss.forward(logits_lb.cuda(), mask=1, label=labels, supervised=True)

				# Get mask and pseudo-labels
				logits_ulb_w = logits_ulb_w.detach()
				logits_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
				max_probsulb_w, max_idxulb_w = torch.max(logits_ulb_w, dim=-1)  # weak
				mask = max_probsulb_w.ge(putoff).long()
				pseudo_lb = max_idxulb_w.long()

				if mask.sum() == 0:
					nloss2 = 0
				else:
					nloss2, _ = self.speaker_loss.forward(logits_ulb_s.cuda(), mask=mask, label=pseudo_lb, supervised=False)

				# Calculate the quality and quantity of pseudo-labels
				quality += (pseudo_lb.eq(y_ulb) * mask).float().sum()
				quantity += mask.sum()

				minprob += max_probsulb_w[mask == 0].sum()  # calculate minimum threshold
				minindex += len(y_ulb) - mask.sum()

				nloss = nloss1 + nloss2
				nloss.backward()
				time_used = time.time() - tstart
				self.optim.step()
				index += len(labels)
				indexulb += len(y_ulb)
				top1 += prec
				loss += nloss.detach().cpu().numpy()
				sys.stderr.write(time.strftime("%H:%M:%S") + \
								 "[%2d]Lr:%5f,Lb:%.2f%%[%.2f],Ulb:%.2f%%" % (
								 epoch, lr, 100 * (num / loader1.__len__()), time_used * loader2.__len__() / num / 60,
								 100 * (num / loader2.__len__())) + \
								 "Ls:%.3f, ACC:%2.2f%%, quality:%.4f,quantity:%.4f\r" % (
								 loss / (num), top1 / index * len(labels),
								 quality / quantity, quantity / indexulb))
				num += 1
				sys.stderr.flush()
			sys.stdout.write("\n")
			gc.collect()
			torch.cuda.empty_cache()
			return loss / num, lr, top1 / index * len(labels), minprob / minindex, quality / quantity, quantity / indexulb

		else:   # Stage2
			for (idx, data, labels) in loader3:
				self.zero_grad()
				labels = torch.LongTensor(labels).cuda()

				speaker_embedding, logits_lb = self.speaker_encoder.forward(data.cuda(), aug=True)
				nloss1, prec = self.speaker_loss.forward(logits_lb.cuda(), mask=0, label=labels)

				nloss1.backward()
				time_used = time.time() - tstart
				self.optim.step()
				index += len(labels)
				top1 += prec
				loss += nloss1.detach().cpu().numpy()
				sys.stderr.write(time.strftime("%H:%M:%S") + "[%2d]Lr:%5f,Lb:%.2f%%[%.2f]" % (epoch, lr, 100 * (num / loader3.__len__()), time_used * loader3.__len__() / num / 60) + \
				"Ls:%.3f,ACC:%2.2f%%\r" % (loss / num, top1/index*len(labels)))
				sys.stderr.flush()
				num += 1
			sys.stdout.write("\n")
			gc.collect()
			torch.cuda.empty_cache()
			return loss / num, lr, top1 / index * len(labels), 0, 0, 0


	# Calculate the initial threshold
	@torch.no_grad()
	def getinitp(self, loader):
		self.eval()
		index, k = 0, 0
		for (idx, data, labels) in tqdm.tqdm(loader):
			with torch.no_grad():
				speaker_embedding, logits_lb = self.speaker_encoder.forward(data.cuda(), aug=True)
				logits_lb = logits_lb.detach()
				logits_lb = torch.softmax(logits_lb, dim=-1)
				max_probslb, max_idxlb = torch.max(logits_lb, dim=-1)
				select = max_idxlb.eq(labels.cuda()).float()
				index += select.sum()
				k += max_probslb[select == 1].sum()
		return k / index

	# Check that the model is performing optimally
	def check(self, EERs, EERm):
		check = False
		if len(EERm) > 2 and EERm[-1] > min(EERs) and EERm[-2] > min(EERs) and EERm[-3] > min(EERs):
			check = True
		return check

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
			audio, _ = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis=0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1, _ = self.speaker_encoder.forward(data_1, aug=False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2, _ = self.speaker_encoder.forward(data_2, aug=False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("Network.", "speaker_encoder.")
				if name not in self_state:
					#print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)