import argparse
import csv
import os
import random
import sys
from datetime import datetime
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel

import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
import numpy as np
from sklearn.metrics import confusion_matrix

import flops_benchmark
from clr import CyclicLR
from data import get_loaders, get_transform
from logger import CsvLogger
from model import ShuffleNetV2
from run import FocalLoss, correct, save_checkpoint, find_bounds_clr

from tqdm import tqdm, trange

epoch_scale = 2
class_nums = 2
img_size = 224
data_val_path = "./orig_data/sort_val/"
c_tag = 0.5
batch_size = 128 * 8

parser = argparse.ArgumentParser(description='ShuffleNetv2 training with PyTorch')
parser.add_argument('--dataroot', metavar='PATH', default='./data/', 
                    help='Path to ImageNet train and val folders, preprocessed as described in '
                         'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default="0", help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Optimization optionss
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=batch_size, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.01, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60 / epoch_scale, 120 / epoch_scale, 240 / epoch_scale, 480 / epoch_scale, 960 / epoch_scale],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=0.1, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=50,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpointss
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')


def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
    model.train()
    correct1, correct2 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        corr = correct(output, target, topk=(1, 2))
        correct1 += corr[0]
        correct2 += corr[1]

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}. '
                'Top-1 accuracy: {:.4f}%({:.4f}%). '
                'Top-2 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct2 / (batch_size * (batch_idx + 1))))
    return loss.item(), correct1 / len(loader.dataset), correct2 / len(loader.dataset)

def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct2 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 2))
        correct1 += corr[0]
        correct2 += corr[1]

    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.10f}, Top1: {}/{} ({:.4f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct2),
                                       len(loader.dataset), 100. * correct2 / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct2 / len(loader.dataset)

def val(model, loader, device, dtype):
    model.eval()
    preds_list = []
    label_list = []
    max_fake_score = 0
    min_real_score = 1
    real_tab = 1
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device = device, dtype = dtype)
        target = target.to(device = device)

        with torch.no_grad():
            output = model(data)
            acc_res = correct(output, target, topk=(1, 2))
            
            output_soft = torch.softmax(output, dim = -1)[ : , 1]

            preds = output_soft.to(device).detach().cpu().numpy()
            labels = target.to(device).detach().cpu().numpy()


            for n in range(len(preds)):
                labels[n] = (labels[n] == real_tab)
                if labels[n] != real_tab:
                    if preds[n] > max_fake_score:
                        max_fake_score = preds[n]
                else:
                    if preds[n] < min_real_score:
                        min_real_score = preds[n]

            preds_list.extend(preds)
            label_list.extend(labels)
         
            print("val process: " + str(batch_idx + 1) + "/" + str(len(loader)))
    
    for n in range(len(preds_list)):
        # if label_list[n] == real_tab:
        #     preds_list[n] = 1 if preds_list[n] > max_fake_score else 0
        # else:
        #     preds_list[n] = 1 if preds_list[n] > min_real_score else 0
        
        preds_list[n] = 1 if preds_list[n] > max_fake_score else 0

    print("max_fake_score: " + str(max_fake_score))
    print("min_real_score: " + str(min_real_score))
    print(confusion_matrix(label_list, preds_list))
    tn, fp, fn, tp = confusion_matrix(label_list, preds_list).ravel()

    apcer = fn / (tp + fn + 1)
    bpcer = fp / (tn + fp + 1)
    acer = (apcer + bpcer) / 2

    print("Apcer: " + str(round(apcer, 6)))
    print("Bpcer: " + str(round(bpcer, 6)))
    print("Acer: "  + str(round(acer, 6)))

    return acer, bpcer, apcer

def train_network(start_epoch, epochs, scheduler, model, train_loader, test_loader, val_loader, optimizer, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test, best_loss):
    
    best_acer, best_bpcer, best_apcer = val(model,val_loader, device, dtype)
    for epoch in trange(start_epoch, epochs + 1):

        start = time.time()
        train_loss, train_acc, train_acc_last, = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval, scheduler)
        test_loss, test_acc, test_acc_last = test(model, test_loader, criterion, device, dtype)
        end = time.time()
        print(str(test_acc) + " vs " + str(best_test))

        if test_acc >= best_test:
            best_test = test_acc
            acer, bpcer, apcer = val(model, val_loader, device, dtype)

            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, acer <= best_acer, filepath=save_path)

            if acer <= best_acer:
                best_acer = acer
                best_bpcer = bpcer
                best_apcer = apcer

        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_acc, 'val_error5': 1 - test_acc_last,
                          'val_loss': test_loss, 'train_error1': 1 - train_acc,
                          'train_error5': 1 - train_acc_last, 'train_loss': train_loss})
        
        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)
        print('epoch: ' + str(epoch + 1)
            + '||test_error: ' + str(round(1 - test_acc, 4))
            + '||test_loss: ' + str(round(test_loss, 4))
            + '||train_error: ' + str(round(1 - train_acc, 4))
            + '||train_loss: ' +  str(round(train_loss, 4))
            + '||best_acer: ' + str(round(best_acer, 4))
            + '||best_bpcer: ' + str(round(best_bpcer, 4))
            + '||best_apcer: ' +  str(round(best_apcer, 4))
            + '||cost time: ' + str(round(end - start, 4)) + " s\n")

        if not isinstance(scheduler, CyclicLR):
            scheduler.step()

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))

def main():
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(10000, 100000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    stages_out_channels = [24, 48, 96, 192, 1024]
    if c_tag == 1:
        stages_out_channels = [24, 116, 232, 464, 1024]
    elif c_tag == 1.5:
        stages_out_channels = [24, 176, 352, 704, 1024]
    elif c_tag == 2:
        stages_out_channels = [24, 244, 488, 976, 2048]

    stages_repeats = [4, 8, 4]

    model = ShuffleNetV2(stages_repeats = stages_repeats, stages_out_channels = stages_out_channels, num_classes=class_nums)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    # print(model)
    print('number of parameters: {}'.format(num_parameters))
    print('FLOPs: {}'.format(
        flops_benchmark.count_flops(ShuffleNetV2,
                                    args.batch_size // len(args.gpus) if args.gpus is not None else args.batch_size,
                                    device, dtype, img_size, 3, stages_repeats,  stages_out_channels)))

    train_loader, test_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, img_size,
                                           args.workers, data_val_path)

    # define loss function (criterion) and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(class_num = class_nums)

    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    if args.find_clr:
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=save_path)
        return

    if args.clr:
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    # optionally resume from a checkpoint
    data = None
    best_test = 0.9
    best_loss = 0.1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = args.start_epoch
            print(checkpoint.keys())

            best_test = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = args.start_epoch

            best_test = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        loss, top1, top5 = test(model, test_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None

    train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, test_loader, val_loader, optimizer, criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                  best_test, best_loss)

if __name__ == '__main__':
    main()
