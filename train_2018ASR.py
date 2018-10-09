import sys
from termcolor import colored, cprint
from torch.autograd import Variable
import torch.utils.data as Data
import torch
import torch.nn as nn
import tensorflow as tf
import os
import numpy as np
import time
from random import randint
# Github Source. for CTC. 
#from decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
from model_2018ASR import DeepSpeech
from decoder_2018ASR import GreedyDecoder
# IMPORTANT
# Insert a blank label
phn =  ['_','B', 'D', 'E', 'G', 'H', 'N', 'S', 'U', 'Wi', 'Z',
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'jE', 'ja', 
    'je', 'jo', 'ju', 'jv', 'k', 'm', 'n', 'o', 'p', 'r', 
    's', 't', 'u', 'v', 'wE', 'wa', 'we', 'wi', 'wv', 'xb',
    'xd', 'xg', 'xl', 'xm', 'xn', 'z']
# 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CSV_saver(object):
    def __init__(self,ex_name,csv_reset=True):
        self.ex_name = ex_name
        self.csv_reset = csv_reset

        self.result_path,self.save_path = self.path()
        self.fd_loss, self.fd_val_result, self.fd_val_acc = self.save_to()

    def path(self):
        # make save and result path
        result_path = os.path.join('./results', self.ex_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        save_path = os.path.join('./save', self.ex_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return result_path, save_path

    def get_path(self):
        # get save and result path.
        return self.result_path, self.save_path

    def save_to(self):
        # make csv file and open it.
        loss = self.result_path + '/loss.csv'
        val_result = self.result_path + '/val_result.csv'
        val_acc = self.result_path + '/val_acc.csv'
        if self.csv_reset:
            fd_val_result = open(val_result, 'a')
            fd_loss = open(loss, 'a')
            fd_val_acc = open(val_acc, 'a') # not overwrite
            fd_val_result.write('continue learning\n')
            fd_loss.write('continue learning\n')
            fd_val_acc.write('continue learning\n')
        else:
            print('**** Remove the csv files ****')
            if os.path.exists(val_acc):
                os.remove(val_acc)
            if os.path.exists(loss):
                os.remove(loss)
            if os.path.exists(val_result):
                os.remove(val_result)
            fd_loss = open(loss,'w')
            fd_val_result = open(val_result,'w')
            fd_val_acc = open(val_acc,'w')
            fd_val_result.write('step,val_result\n')
            fd_loss.write('step,loss\n')
            fd_val_acc.write('step,val_acc\n')
        return fd_loss, fd_val_result, fd_val_acc
    
    def write(self,step,data,file):
        # write values in csv file.
        if file == 'loss':
            self.fd_loss.write(str(step) + ',' + str(data) + "\n")
            self.fd_loss.flush()
        elif file == 'val_result':
            # wirte real and decoded sentence.
            decoded_sentence, real_sentence = data
            # decoded_sentence and real_sentence should be str.
            self.fd_val_result.write(str(step)+"\n"+ decoded_sentence + '\n' + real_sentence + "\n")
            self.fd_val_result.flush()
        elif file == 'val_acc':
            self.fd_val_acc.write(str(step) + ',' + str(data) + "\n")
            self.fd_val_acc.flush()
        else:
            raise ValueError

    def close(self):
        # close the csv file at the end of code.
        self.fd_loss.close()
        self.fd_val_result.close()
        self.fd_val_acc.close()

def data_load(path, is_training, batch_size, num_workers=2, mode='fbank'):
    # data_path = 
    # xPath: 
    # 1. PATH
    keyword = ['Keyword','Nonkeyword']
    noise = ['TV','냉장고']
    # 1.             
    xPath = []
    yPath = []
    for keyword_ in keyword:
        for noise_ in noise:
            xPath.append(os.path.join(path,mode,keyword_,is_training,noise_))
            yPath.append(os.path.join(path,'label',keyword_,is_training,noise_))
    print(xPath)
    
    # 1. 
    xList = []
    yList = []
    total_num = 0
    assert len(xPath) == len(yPath)
    for i in range(len(xPath)):
        xList += [np.load(os.path.join(xPath[i], fn)) for fn in os.listdir(xPath[i])]
        yList += [np.load(os.path.join(yPath[i], fn)) for fn in os.listdir(yPath[i])]
        total_num += len(os.listdir(xPath[i]))
    assert len(xList) == len(yList) #4620
    cprint('total number of train data: '+str(total_num), 'green')
    cprint(xList[0].shape, 'green')
    cprint(yList[0].shape, 'green')

    dataset = []
    max_target = 0

    for x in range(len(xList)):
        inputs = torch.from_numpy(xList[x]).type(torch.FloatTensor)
        targets = torch.from_numpy(yList[x]).type(torch.IntTensor)
        if torch.max(targets) > max_target:
            max_target = torch.max(targets)
        dataset.append((inputs,yList[x].tolist()))
    cprint('Max target: '+str(max_target),'green')
    def _collate_fn(batch):
        def func(p):
            return p[0].size(1)
        ### batch: list
        ### batch[0] : tuple: (tensor([2darray],[1darray]))
        ### batch[0][0] : torch.FloatTensor
        ### batch[0][1] : list
        longest_sample = max(batch, key=func)[0]
        freq_size = longest_sample.size(0)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(1)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)
        target_sizes = torch.IntTensor(minibatch_size)
        targets = []
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(target)
            targets.extend(target)
        targets = torch.IntTensor(targets)
        return inputs, targets, input_percentages, target_sizes

    return Data.DataLoader(dataset, batch_size=batch_size,collate_fn = _collate_fn,num_workers=num_workers,
        shuffle=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='2018_spring_Speech Recognition system_final project_Keyword Spotting.')
    #PATH
    parser.add_argument('--data_path', default='./feature_saved', type=str)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--ex_name', default='noname', type=str)
    parser.add_argument('--continue_from', default=0, type=int)
    parser.add_argument('--SampleTest', default=0, type=int)
    # Experiment
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cuda', default=2, type=int)
    parser.add_argument('--epoch', default=100, type=int)

    parser.add_argument('--optimizer', default='Adam', type=str,choices=['SGD','Adam'], help='optimizer choose')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    args = parser.parse_args()
    # 
    # 1. Load data and make data loader.
    #
    if args.continue_from:
    	print('*'*10 + 'continue from: ' + str(args.continue_from) + '*'*10)
    else:
        print('*'*10 + 'new training'+'*'*10)
    if args.SampleTest:
        train_loader = data_load(path=args.data_path, is_training='Sample', batch_size=args.batch_size)
        test_loader = data_load(path=args.data_path, is_training='Sample', batch_size=args.batch_size)
    else:
        train_loader = data_load(path=args.data_path, is_training='Train', batch_size=args.batch_size)
        test_loader = data_load(path=args.data_path, is_training='Test', batch_size=args.batch_size)

    # 1. Setting.
    CSV_saver = CSV_saver(args.ex_name,args.continue_from)
    _, save_path = CSV_saver.get_path()
    avg_loss, start_epoch, start_iter = 0, 0, 0
    # loss function: ctc loss.
    criterion = CTCLoss()
    labels = phn
    decoder = GreedyDecoder(labels)

    if args.continue_from:
        load_model_path = save_path + '/model_checkpoint_' + str(args.continue_from) + '.pth'
        package = torch.load(load_model_path, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = DeepSpeech.get_labels(model)
        parameters = model.parameters()
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        optimizer.load_state_dict(package['optim_dict'])

        # Temporary fix for pytorch #2830 & #1442 while pull request #3658 in not incorporated in a release
        # TODO : remove when a new release of pytorch include pull request #3658
        if args.cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))
        loss_results, per_results = package['loss_results'], package['per_results']
        #best_per = package['best_per']
    else:
        model = DeepSpeech(rnn_hidden_size=800,
                           nb_layers=5,
                           labels=labels,
                           rnn_type=nn.LSTM,
                           audio_conf=None,
                           bidirectional=True)

        parameters = model.parameters()
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        best_per,best_per_epoch = None,0
    # CUDA SETTING
    if args.cuda:
        #model = torch.nn.DataParallel(model,device_ids=[0,1,2]).cuda()
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    
    # SHOULD CONSIDER USING TENSORFLOW DECODER.
    #decoder = GreedyDecoder(labels)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_results, per_results = torch.Tensor(args.epoch), torch.Tensor(args.epoch)


    for epoch in range(start_epoch, start_epoch+args.epoch):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        avg_loss = 0
        for idx, data in enumerate(train_loader):
            inputs, targets, input_percentages,target_sizes=data
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False) #[10,1,120,423] [N,1,feature,seqlen]
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            if args.cuda:
                inputs = inputs.cuda()
            out = model(inputs)
            out = out.transpose(0,1) # [Time, N, H]
            #print(out.size(), len(labels))
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            loss = criterion(out,targets,sizes,target_sizes)
            loss = loss / inputs.size(0)

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            #
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()
            
            if args.cuda:
                torch.cuda.synchronize()
            # time compute
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (idx + 1), len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        avg_loss /= len(train_loader)
        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, epoch_time=epoch_time, loss=avg_loss))
        
        # VALIDATION EPOCH START
        total_cer, total_per = 0, 0
        model.eval()
        for idx, data in enumerate(test_loader):
            inputs, targets, input_percentages,target_sizes = data
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False) #[10,1,120,423] [N,1,feature,seqlen]

            # UNFLATTEN TARGETS
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
            
            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs) # N,T,H
            seq_length = out.size(1)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output, _ = decoder.decode(out.data, sizes)
            target_strings = decoder.convert_to_strings(split_targets)
            per = 0
            random_pick = randint(0,len(target_strings)-1)
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                per += decoder.wer(transcript, reference) / float(len(reference))
                if x == random_pick:
                    CSV_saver.write(epoch+1,(transcript,reference),'val_result')        
            total_per += per

            if args.cuda:
                torch.cuda.synchronize()
            del out

        per = total_per / len(test_loader.dataset)
        per *= 100
        loss_results[epoch] = avg_loss
        per_results[epoch] = per
        print('Validation Summary Epoch: [{0}]\t'
              'Average per {per:.3f}\t'.format(
            epoch + 1, per=per))
        # CSV SAVE
        CSV_saver.write(epoch+1,float(avg_loss),'loss')
        CSV_saver.write(epoch+1,float(per),'val_acc')

        if (best_per is None or best_per > per):
            model_path = save_path + '/model_checkpoint_' + str(epoch+1) + '.pth'
            remove_path = save_path + '/model_checkpoint_' + str(best_per_epoch) + '.pth'
            cprint('best_per: {}, current per: {}'.format(best_per, per),'yellow')
            cprint("Found better validated model, saving to %s" % model_path,'yellow')           
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, 
                        loss_results=loss_results, per_results=per_results, best_per= best_per), model_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)
                cprint("Remove %s" % remove_path,'yellow')           
            best_per = per
            best_per_epoch = epoch+1
    CSV_saver.close()
    