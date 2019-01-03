from common import *
from data import HumanDataset
from kaggle_human_protein_baseline.model import*


import argparse
#-----------------------------------------------
# Arg parser
# changes
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME", help="NAME OF OUTPUT FOLDER",
                    type=str)
parser.add_argument("--INITIAL_CHECKPOINT", help="CHECK POINT",
                    type=str)
parser.add_argument("--RESUME", help="RESUME RUN",
                    type=bool)
parser.add_argument("--BATCH_SIZE", help="BATCH SIZE TIMES NUMBER OF GPUS",
                    type=int)
parser.add_argument("--MODE", help="TRAIN OR TEST",
                    type=str)
args = parser.parse_args()

config.resume = args.RESUME
config.model_name = args.MODEL_NAME
config.initial_checkpoint = args.INITIAL_CHECKPOINT
config.batch_size = args.BATCH_SIZE
config.mode = args.MODE


# 1. set random seed
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
try:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
    NUM_CUDA_DEVICES = 1
warnings.filterwarnings('ignore')

if not os.path.exists(config.logs):
    os.mkdir(config.logs)

log = Logger()
log.open('{0}{1}_log_train.txt'.format(config.logs, config.model_name),mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

# tflogger
tflogger = TFLogger(os.path.join('results','TFlogs', config.model_name))

def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start, threshold=0.3):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = data_parallel(model,images)
        # output = model(images)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))
        
        f1_batch = f1_score(target.cpu(),output.sigmoid().cpu() > threshold,average='macro')
        f1.update(f1_batch,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f           |         %s      %s            | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg, 
                valid_loss[0], valid_loss[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [losses.avg,f1.avg]

# 2. evaluate fuunction
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start, threshold=0.3):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
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

        # compute loss for the entire evaluation dataset
        loss = criterion(total_output, total_target)
        losses.update(loss.item(), images_var.size(0))
        f1_batch = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > threshold, average='macro')
        f1.update(f1_batch, images_var.size(0))
        print('\r', end='', flush=True)
        message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % ( \
            "val", epoch, epoch,
            train_loss[0], train_loss[1],
            losses.avg, f1.avg,
            str(best_results[0])[:8], str(best_results[1])[:8],
            time_to_str((timer() - start), 'min'))

        print(message, end='', flush=True)

        log.write("\n")

    return [losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv("./input/sample_submission.csv")
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            #print(label > 0.5)
           
            labels.append(label > 0.15)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./results/submit/%s_bestloss_submission.csv'%config.model_name, index=None)

# 4. main function
def main():

    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep + 'fold_'+str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep + 'fold_'+ str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

    # 4.2 get model
    model = get_net()
    model.cuda()

    # -------------------------------------------------------
    # training
    # -------------------------------------------------------
    if config.mode == 'train':
        # criterion
        optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
        # criterion = nn.BCEWithLogitsLoss().cuda()
        # criterion = FocalLoss().cuda()
        criterion = F1_loss().cuda()
        best_loss = 999
        best_f1 = 0
        best_results = [np.inf,0]
        val_metrics = [np.inf,0]

        all_files = pd.read_csv("./input/train.csv")
        #print(all_files)
        train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)

        # load dataset
        train_gen = HumanDataset(train_data_list,config.train_data,mode="train")
        train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=4)

        val_gen = HumanDataset(val_data_list,config.train_data,augument=False,mode="train")
        val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=4)

        if config.resume:
            log.write('\tinitial_checkpoint = %s\n' % config.initial_checkpoint)
            checkpoint_path = os.path.join(config.weights, config.model_name, config.initial_checkpoint,'checkpoint.pth.tar')
            loaded_model = torch.load(checkpoint_path)
            model.load_state_dict(loaded_model["state_dict"])
            start_epoch = loaded_model["epoch"]
        else:
            start_epoch = 0

        scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
        start = timer()

        #train
        for epoch in range(start_epoch,config.epochs):
            scheduler.step(epoch)
            # train
            lr = get_learning_rate(optimizer)
            train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start,config.threshold)
            # val
            val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start, config.threshold)
            # check results
            is_best_loss = val_metrics[0] < best_results[0]
            best_results[0] = min(val_metrics[0],best_results[0])
            is_best_f1 = val_metrics[1] > best_results[1]
            best_results[1] = max(val_metrics[1],best_results[1])
            # save model
            save_checkpoint({
                        "epoch":epoch + 1,
                        "model_name":config.model_name,
                        "state_dict":model.state_dict(),
                        "best_loss":best_results[0],
                        "optimizer":optimizer.state_dict(),
                        "fold":fold,
                        "best_f1":best_results[1],
            },is_best_loss,is_best_f1,fold)
            # print logs
            print('\r',end='',flush=True)
            log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "best", epoch, epoch,
                    train_metrics[0], train_metrics[1],
                    val_metrics[0], val_metrics[1],
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
                )
            log.write("\n")
            time.sleep(0.01)

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = {'Train_loss': train_metrics[0], 'Train_F1_macro': train_metrics[1],
                    'Valid_loss': val_metrics[0], 'Valid_F1_macro': val_metrics[1]}

            for tag, value in info.items():
                tflogger.scalar_summary(tag, value, epoch)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                tflogger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                tflogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
            # -------------------------------------
            # end tflogger

    # -------------------------------------------------------
    # testing
    # -------------------------------------------------------
    elif config.mode=='test':
        test_files = pd.read_csv("./input/sample_submission.csv")
        test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
        test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

        checkpoint_path = os.path.join(config.best_models,'{0}_fold_{1}_model_best_loss.pth.tar'.format(config.model_name, fold))
        best_model = torch.load(checkpoint_path)
        #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
        model.load_state_dict(best_model["state_dict"])
        test(test_loader,model,fold)
        print('Test successful!')
if __name__ == "__main__":
    main()
