'''
This is the code of TSF-ASE-Stage1, which is modified from the Stage 1 of LGL : https://github.com/TaoRuijie/Loss-Gated-Learning
'''
import sys, time, os, argparse, warnings, glob, torch
from tools import *
from model import *
from encoder import *
from dataLoader import *
import yaml

def main(args):
    '''
    main_worker is conducted on each GPU.
    '''
    # Initialization
    model_save_path     = args.save_path+"/model"
    os.makedirs(model_save_path, exist_ok = True)
    scorefile = open(args.save_path+"/scores.txt", "a+")
    it = 1

    Trainer = model(lr=args.lr,lr_decay=args.lr_decay) # Define the framework
    modelfiles = glob.glob('%s/model0*.model'%model_save_path) # Search the existed model files
    modelfiles.sort()

    if(args.initial_model != ""): # If initial_model is exist, system will train from the initial_model
        Trainer.load_network(args.initial_model)
    elif len(modelfiles) >= 1: # Otherwise, system will try to start from the saved model&epoch
        Trainer.load_network(modelfiles[-1])
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    if args.eval == True: # Do evaluation only
        EER, minDCF = Trainer.evaluate_network(**vars(args))
        print('EER %2.4f, minDCF %.3f\n'%(EER, minDCF))
        quit()

    trainLoader = get_loader(args) # Define the dataloader

    while it < args.max_epoch:
        # Train for one epoch
        loss, traineer, lr = Trainer.train_network(loader=trainLoader, epoch = it)

        # Evaluation every [test_interval] epochs, record the training loss, training acc, evaluation EER/minDCF
        if it % args.test_interval == 0:
            Trainer.save_network(model_save_path+"/model%09d.model"%it)
            EER, minDCF = Trainer.evaluate_network(**vars(args))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, Acc %2.2f, LOSS %f, EER %2.4f, minDCF %.3f"%( lr, traineer, loss, EER, minDCF))
            scorefile.write("Epoch %d, LR %f, Acc %2.2f, LOSS %f, EER %2.4f, minDCF %.3f\n"%(it, lr, traineer, loss, EER, minDCF))
            scorefile.flush()
        # Otherwise, recored the training loss and acc
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, Acc %2.2f, LOSS %f"%( lr, traineer, loss))
            scorefile.write("Epoch %d, LR %f, Acc %2.2f, LOSS %f\n"%(it, lr, traineer, loss))
            scorefile.flush()

        it += 1
        print("")

def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Stage I, self-supervsied speaker recognition with contrastive learning.")
    parser.add_argument('--max_frames', type=int, default=180, help='Input length to the network, 1.8s')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size, bigger is better')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
    parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch', type=int, default=60, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs')
    parser.add_argument('--initial_model', type=str, default="", help='Initial model path')
    parser.add_argument('--save_path', type=str, default="/home/ljh/exps/contrast", help='Path for model and scores.txt')
    parser.add_argument('--train_list', type=str, default="/home/ljh/data/speaker/voxceleb2/train_list.txt", help='Path for Vox2 list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
    parser.add_argument('--val_list', type=str, default="/home/ljh/data/speaker/voxceleb1/veri_test2.txt",help='Path for Vox_O list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--train_path', type=str, default="/home/ljh/data/speaker/voxceleb2/train/wav", help='Path to the Vox2 set')
    parser.add_argument('--val_path', type=str, default="/home/ljh/data/speaker/voxceleb1/test/wav", help='Path to the Vox_O set')
    parser.add_argument('--musan_path', type=str, default="/home/ljh/data/speaker/Others/musan_split", help='Path to the musan set')
    parser.add_argument('--eval', dest='eval', action='store_true', help='Do evaluation only')

    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    warnings.filterwarnings("ignore")
    over_write_args_from_file(args, args.c)
    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    if not torch.cuda.is_available():
        raise Exception('ONLY CPU TRAINING IS SUPPORTED')
    else:
        print('Number of GPUs:', torch.cuda.device_count())
        print('Save path:', args.save_path)
        if n_gpus == 1:
            main(args)