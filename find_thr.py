import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
from utils import *
from data import HumanDataset
# from common import *
from kaggle_human_protein_baseline.model import*


import argparse
#-----------------------------------------------
# Arg parser
# changes
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME", help="NAME OF OUTPUT FOLDER",
                    type=str)
parser.add_argument("--BATCH_SIZE", help="BATCH SIZE TIMES NUMBER OF GPUS",
                    type=int)
parser.add_argument("--GPUS", help="GPU",
                    type=str)
parser.add_argument("--CHECKPOINT", help="CHECK POINT FOR TEST",
                    default=0, type=int)

args = parser.parse_args()


config.model_name = args.MODEL_NAME

config.batch_size = args.BATCH_SIZE

config.gpus = args.GPUS

config.checkpoint = args.CHECKPOINT


# 1. set random seed
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
try:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
    NUM_CUDA_DEVICES = 1
warnings.filterwarnings('ignore')

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }



def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.0,d=25.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = np.zeros(len(name_label_dict))
    wd = 1e-5

    # This function is trying to minimize (F1-1)^2 + wd^2*th^2.
    # So, you are trying to get F1 close to 1. while if something goes wrong,
    # like there are just a few data points, thresholds do not go far away from 0.0. Also,
    # I'm using a soft version of F1 to have stable gradient in least square minimization.
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*p), axis=None)
    p, success = opt.leastsq(error, params)
    return p

# validate
def validate(val_loader,model):
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(tqdm(val_loader)):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

            # optain output of a batch
            output = data_parallel(model, images_var)

            # Concatenate all every batch
            if i == 0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output], 0)
                total_target = torch.cat([total_target, target], 0)

    return [total_output, total_target]

if __name__ == '__main__':
    all_files = pd.read_csv("./input/train.csv")
    # print(all_files)
    # train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)

    # using a split that includes all classes in val
    with open(os.path.join("./input/protein-trainval-split", 'tr_names.txt'), 'r') as text_file:
        train_names = text_file.read().split(',')
        # oversample
        s = Oversampling("./input/train.csv")
        train_names = [idx for idx in train_names for _ in range(s.get(idx))]
        # train_data_list = all_files[all_files['Id'].isin(train_names)]
        train_data_list = all_files.copy().set_index('Id')
        # train_data_list
        train_data_list = train_data_list.reindex(train_names)
        # 57150 -> 29016
        # reset index
        train_data_list = train_data_list.rename_axis('Id').reset_index()
    with open(os.path.join("./input/protein-trainval-split", 'val_names.txt'), 'r') as text_file:
        val_names = text_file.read().split(',')
        val_data_list = all_files[all_files['Id'].isin(val_names)]

    # 4.2 get model
    model = get_net()
    model.cuda()
    fold = 0

    # load dataset
    train_gen = HumanDataset(train_data_list, config.train_data, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    val_gen = HumanDataset(val_data_list, config.train_data, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)


    # checkpoint_path = os.path.join(config.best_models,'{0}_fold_{1}_model_best_loss.pth.tar'.format(config.model_name, fold))
    checkpoint_path = os.path.join(config.weights, config.model_name, 'fold_{0}'.format(fold),
                                   'checkpoint_{}.pth.tar'.format(config.checkpoint))
    best_model = torch.load(checkpoint_path)
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])


    preds,y = validate(train_loader,model)
    preds = preds.cpu()
    y = y.cpu()
    y = np.array(y)

    preds = np.stack(preds, axis=-1)
    pred = preds.mean(axis=-1)



    th = fit_val(pred,y)
    print('Thresholds: ',th)
    print('F1 macro: ',f1_score(y, pred>th, average='macro'))
    print('F1 macro (th = 0.0): ',f1_score(y, pred>0.0, average='macro'))
    print('F1 micro: ',f1_score(y, pred>th, average='micro'))

