import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
from utils import *
from data import HumanDataset
# from common import *
from kaggle_human_protein_baseline.model import*
from sklearn.model_selection import train_test_split

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


    preds,y = validate(val_loader,model)
    preds = preds.cpu()
    preds = np.array(preds)
    y = y.cpu()
    y = np.array(y)

    # preds = np.stack(preds, axis=-1)
    # pred = preds.mean(axis=-1)


#
#     th = fit_val(preds,y)
#     print('Thresholds: ',th)
#     print('F1 macro: ',f1_score(y, preds>th, average='macro'))
#     print('F1 macro (th = 0.0): ',f1_score(y, preds>0.0, average='macro'))
#     print('F1 micro: ',f1_score(y, preds>th, average='micro'))
#
# # Thresholds:  [-0.79494995 -0.08028803 -0.06839066 -0.54636391  0.39274047 -0.06704237
# #  -0.8016432  -0.46357643  0.43253807 -0.84245714 -2.84211516 -0.27193429
# #   0.13533374 -0.16440481 -0.1186454   0.          0.15110027 -0.18069459
# #  -0.4793194  -0.50146258 -0.2156128  -0.31305433 -1.46252153 -1.41710544
# #  -0.34938362 -0.13460358  0.07340712  0.        ]
# # F1 macro:  0.6668250095431967
# # F1 macro (th = 0.0):  0.6476885711723936
# # F1 micro:  0.7466047591410332
#
#
#
# # Using CV to prevent overfitting the thresholds:
th, score, cv = 0.5,0,10
for i in range(cv):
    xt,xv,yt,yv = train_test_split(preds,y,test_size=0.5,random_state=i)
    th_i = fit_val(xt,yt)
    th += th_i
    score += f1_score(yv, xv>th_i, average='macro')
th/=cv
score/=cv
print('Thresholds: ',th)
print('F1 macro avr:',score)
print('F1 macro: ',f1_score(y, preds>th, average='macro'))
print('F1 micro: ',f1_score(y, preds>th, average='micro'))
# Thresholds:  [-0.27631527 -0.31156957 -0.61893745 -1.01863398 -0.3141709  -0.14000374
#  -0.6285302  -0.43241383 -1.60594984 -0.14425374 -0.03979607 -0.25717957
#  -0.84905692 -0.37668712  1.3710663  -0.11193908 -0.81109447  0.72506607
#  -0.05454339 -0.47056617 -0.16024197 -0.44002794 -0.65929407 -1.00900269
#  -0.86197429 -0.12346229 -0.4946575  -0.52420557]
# F1 macro:  0.8474219916912414
# F1 macro (th = 0.0):  0.8318519970045825
# F1 micro:  0.8728054927863723


# thresholds = [-0.13432257, -0.4642075, -0.50726506, -0.49715518, -0.41125674, 0.11581507,
#               -1.0143597, -0.18461785, -0.61600877, -0.47275479, -0.9142859, -0.44323673,
#               -0.58404387, -0.22959213, -0.26110631, -0.43723898, -0.97624685, -0.44612319,
#               -0.4492785, -0.56681327, -0.16156543, -0.12577745, -0.75476121, -0.91473052,
#               -0.53361931, -0.19337344, -0.0857145, -0.45739976]

f1 = f1_score(y, preds>th, average=None)
for i in range(len(name_label_dict)):
    bins = np.linspace(preds[:,i].min(), preds[:,i].max(), 50)
    plt.hist(preds[y[:,i] == 0][:,i], bins, alpha=0.5, log=True, label='false')
    plt.hist(preds[y[:,i] == 1][:,i], bins, alpha=0.5, log=True, label='true')
    plt.legend(loc='upper right')
    print(name_label_dict[i],i, f1[i], th[i])
    plt.axvline(x=th[i], color='k', linestyle='--')
    plt.show()

