import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_predict(pred_labels, labels, save_path=''):
    plt.clf()
    ax = plt.gca()
    clean_index = np.where(labels==1)[0]
    noisy_index = np.where(labels==0)[0]
    # Plot data histogram
    ax.hist(pred_labels[clean_index], bins=100, density=True, histtype='stepfilled', color='green', alpha=0.4, label='Clean Pairs')
    ax.hist(pred_labels[noisy_index], bins=100, density=True, histtype='stepfilled', color='red', alpha=0.4, label='Noisy Pairs')

    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def get_correspondence(model, data_loader, epoch,save_path):
    logger = logging.getLogger("IRRA.train")
    model.eval()
    data_size = data_loader.dataset.__len__()
    preds = torch.zeros(data_size)
    labels =  data_loader.dataset.real_correspondences
    uncertainty = torch.zeros(data_size)
    uncertainty1 = torch.zeros(data_size)
    uncertainty2 = torch.zeros(data_size)
    norm_es_eye = torch.zeros(data_size)
    num_loader_iter =   data_size // 64 + 1
    device = "cuda"
    logger.info(f"=> Get predicted correspondence labels at epoch: {epoch}")
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids = batch['index']
        if i % 100 == 0:
            logger.info(f"Eval_train batch: [{i}/{num_loader_iter}], computing evidences")
        
        with torch.no_grad():
            sims = model.forward_sims(batch)
            alpha = (sims/0.1).exp() + 1
            alpha_ = (sims.t()/0.1).exp() + 1
            sum_e = (alpha-1) + (alpha_-1)
            norm_e =  sum_e / torch.sum(sum_e, dim=1, keepdim=True)
            k = sims.size(0)
            g_t = torch.from_numpy(np.array([i for i in range(k)])).cuda()
            pred = g_t.eq(torch.argmax(norm_e, dim=1)) + 0
            u_1 = k / torch.sum(alpha, dim=1, keepdim=True)
            u_2 = k / torch.sum(alpha_, dim=1, keepdim=True)
            u = u_1 + u_2
            for b in range(k):
                preds[ids[b]] = pred[b]
                norm_es_eye[ids[b]] = norm_e[b][b]
                uncertainty[ids[b]] = u[b]
                uncertainty1[ids[b]] = u_1[b]
                uncertainty2[ids[b]] = u_2[b]
    print(norm_es_eye)
    print(uncertainty)
    plot_predict(norm_es_eye, labels, save_path=os.path.join(save_path, 'norm_es_eye{}.pdf'.format(epoch)))
    plot_predict(uncertainty, labels, save_path=os.path.join(save_path, 'uncertainty{}.pdf'.format(epoch)))
    # check the split data
    logger.info('Confusion Matrix by A')
    cm = confusion_matrix(labels, preds+0)
    logger.info('\n'+str(cm))
    logger.info('Noise Recall: {}, Clean Recall: {}'.format(cm[0, 0]/(cm[0, 0]+cm[0, 1]), cm[1, 1]/(cm[1, 0]+cm[1, 1] + 1))) # +1 avoid Nan

    return preds.cpu(), labels, norm_es_eye.cpu(), uncertainty.cpu()


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "hinge_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "loss_rdh": AverageMeter(),
        "loss_edl": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        preds, labels, norm_es_eye, uncertainty =  get_correspondence(model, train_loader, epoch, args.output_dir+'/img/')
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        model.epoch = epoch
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            batch['preds'] = preds[index.cpu()]

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['hinge_loss'].update(ret.get('hinge_loss', 0), batch_size)
            meters['loss_edl'].update(ret.get('loss_edl', 0), batch_size)
            meters['loss_rdh'].update(ret.get('loss_rdh', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    
    
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    
    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)



def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
