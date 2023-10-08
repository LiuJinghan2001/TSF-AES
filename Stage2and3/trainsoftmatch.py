'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, warnings, time, sys
from tools import *
from dataLoader1 import train_loader
from semi_model.softmatch import ECAPAModel
from data_utils import *

def main_worker(args):
	# SET save_path and logger
	save_path = os.path.join(args.save_path, 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	else:
		print('already existing model: {}'.format(save_path))
	print("USE GPU: %s for training" % 0)
	## Define the data loader
	data_list, data_label = get_data(args.train_list, args.train_path)
	lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data_list, data_label,
																lb_samples_per_class=args.lb_samples_per_class,
																num_classes=args.n_class)
	lb_trainloader = train_loader(args, lb_data, lb_targets, args.musan_path, args.rir_path, args.num_frames,
								  labelled=True)
	ulb_trainloader = train_loader(args, ulb_data, ulb_targets, args.musan_path, args.rir_path, args.num_frames,
								   labelled=False)
	Lb_trainloader = torch.utils.data.DataLoader(lb_trainloader, batch_size=args.batch_size, shuffle=True,
											  num_workers=args.n_cpu, drop_last=True)
	Ulb_trainloader = torch.utils.data.DataLoader(ulb_trainloader, batch_size=args.batch_size * args.uratio, shuffle=True,
												 num_workers=args.n_cpu, drop_last=True)
	## Search for the exist models
	modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
	modelfiles.sort()
	## Only do evaluation, the initial_model is necessary
	if args.eval == True:
		s = ECAPAModel(**vars(args))
		print("Model %s loaded from previous state!" % args.initial_model)
		s.load_parameters(args.initial_model)
		EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
		print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
		quit()

	## If initial_model is exist, system will train from the initial_model
	if args.initial_model != "":
		print("Model %s loaded from previous state!" % args.initial_model)
		s = ECAPAModel(**vars(args))
		s.load_parameters(args.initial_model)
		epoch = 1

	## Otherwise, system will try to start from the saved model&epoch
	elif len(modelfiles) >= 1:
		print("Model %s loaded from previous state!" % modelfiles[-1])
		epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
		s = ECAPAModel(**vars(args))
		s.load_parameters(modelfiles[-1])
	## Otherwise, system will train from scratch
	else:
		epoch = 1
		s = ECAPAModel(**vars(args))

	EERs = []
	score_file = open(args.score_save_path, "a+")
	EERm=[]
	device = torch.device("cuda")
	s.cuda()
	epoch = 1

	while (1):

		## Training for one epoch
		loss, lr, acc, quality, quantity = s.train_network(putoff=args.putoff, epoch=epoch, loader1=Lb_trainloader, loader2=Ulb_trainloader)
		## Evaluation every [test_step] epochs

		if epoch % args.test_step == 0:
			s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
			EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
			EERs.append(EER)
			EERm.append(EER)
			print(time.strftime("%Y-%m-%d %H:%M:%S"),
				  "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF%2.2f%%" % (epoch, acc, EERs[-1], min(EERs), minDCF))
			score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF %2.2f%%\n" % (
			epoch, lr, loss, acc, EERs[-1], min(EERs), minDCF))
			score_file.flush()

		if epoch >= args.max_epoch:
			quit()
		epoch += 1


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="ECAPA_TDNN_Trainer")
	## Training Settings
	parser.add_argument('--num_frames', type=int, default=200,
						help='Duration of the input segments, eg: 200 for 2 second')
	parser.add_argument('--max_epoch', type=int, default=53, help='Maximum number of epochs')
	parser.add_argument('--batch_size', type=int, default=150, help='Batch size')
	parser.add_argument('--uratio', type=int, default=1,
						help='the ratio of unlabeled data to labeld data in each mini-batch')
	parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
	parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')
	parser.add_argument("--putoff", type=float, default=0.9, help='threshold')
	parser.add_argument('--train_list', type=str, default="/home/ljh/data/speaker/voxceleb2/train_list.txt",
						help='The path of the training list, eg:"/data08/VoxCeleb2/train_list.txt" in my case, which contains 1092009 lins')
	parser.add_argument('--train_path', type=str, default="/home/ljh/data/speaker/voxceleb2/train/wav",
						help='The path of the training data, eg:"data08/voxceleb2/train/wav" in my case')
	parser.add_argument('--eval_list', type=str, default="/home/ljh/data/speaker/voxceleb1/veri_test2.txt",
						help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
	parser.add_argument('--eval_path', type=str, default="/home/ljh/data/speaker/voxceleb1/test/wav",
						help='The path of the evaluation data, eg:"/data08/voxceleb1/test/wav" in my case')
	parser.add_argument('--musan_path', type=str, default="/home/ljh/data/speaker/Others/musan_split",
						help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
	parser.add_argument('--rir_path', type=str, default="/home/ljh/data/speaker/Others/RIRS_NOISES/simulated_rirs",
						help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case')
	parser.add_argument('--save_path', type=str, default="/home/ljh/exps/test_github",
						help='Path to save the score.txt and models')
	parser.add_argument('--initial_model', type=str, default="",
						help='Path of the initial_model')


	## Model and Loss settings
	parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
	parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
	parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
	parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
	parser.add_argument('--lb_samples_per_class', type=float, default=4, help='Number of speakers')
	## Command
	parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
	parser.add_argument('--environment', type=bool, default=False, help='The environment of training is Windows')

	## Initialization
	warnings.simplefilter("ignore")
	torch.multiprocessing.set_sharing_strategy('file_system')
	args = parser.parse_args()
	args = init_args(args)
	n_gpus = torch.cuda.device_count()

	print('Python Version:', sys.version)
	print('PyTorch Version:', torch.__version__)
	if not torch.cuda.is_available():
		raise Exception('ONLY CPU TRAINING IS SUPPORTED')
	else:
		print('Number of GPUs:', torch.cuda.device_count())
		print('Save path:', args.save_path)
		if n_gpus == 1:
			main_worker(args)

