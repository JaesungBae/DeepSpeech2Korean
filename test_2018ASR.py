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
from tqdm import tqdm
# Github Source. for CTC. 
#from decoder import GreedyDecoder
from model_2018ASR import DeepSpeech
from decoder_2018ASR import GreedyDecoder

phn =  ['_','B', 'D', 'E', 'G', 'H', 'N', 'S', 'U', 'Wi', 'Z',
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'jE', 'ja', 
    'je', 'jo', 'ju', 'jv', 'k', 'm', 'n', 'o', 'p', 'r', 
    's', 't', 'u', 'v', 'wE', 'wa', 'we', 'wi', 'wv', 'xb',
    'xd', 'xg', 'xl', 'xm', 'xn', 'z']

class CSV_saver(object):
    def __init__(self,ex_name,csv_reset=True):
        self.ex_name = ex_name
        self.csv_reset = csv_reset

        self.result_path,self.save_path = self.path()
        self.fd_test_acc, self.fd_test_result = self.save_to()

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
        test_acc = self.result_path + '/test_acc.csv'
        test_result  = self.result_path + '/test_result.csv'
        if self.csv_reset:
            fd_test_result = open(test_result, 'a')
            fd_test_acc = open(test_acc, 'a')
            fd_test_result.write('continue learning\n')
            fd_test_acc.write('continue learning\n')
        else:
            print('**** Remove the csv files ****')
            if os.path.exists(test_result):
                os.remove(test_result)
            if os.path.exists(test_acc):
                os.remove(test_acc)
            fd_test_acc = open(test_acc,'w')
            fd_test_result = open(test_result,'w')
            fd_test_result.write('step,val_result\n')
            fd_test_acc.write('step,loss\n')
        return fd_test_acc, fd_test_result
    
    def write(self,step,data,file):
        # write values in csv file.
        if file == 'test_acc':
            self.fd_test_acc.write(str(step) + ',' + str(data) + "\n")
            self.fd_test_acc.flush()
        elif file == 'test_result':
            # wirte real and decoded sentence.
            decoded_sentence, real_sentence = data
            # decoded_sentence and real_sentence should be str.
            self.fd_test_result.write(str(step)+"\n"+ decoded_sentence + '\n' + real_sentence + "\n")
            self.fd_test_result.flush()
        else:
            raise ValueError

    def close(self):
        # close the csv file at the end of code.
        self.fd_test_acc.close()
        self.fd_test_result.close()
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
        shuffle=False)

def decide_label(transcript,reject_threshold):
    label1 = 'a xl p a b o xd '
    label2 = 'o xn n u r i '
    label3 = 'm i r i n E '
    #
    label1_per = decoder.wer(transcript, label1) / float(len(label1))
    label2_per = decoder.wer(transcript, label2) / float(len(label2))
    label3_per = decoder.wer(transcript, label3) / float(len(label3))
    label_per = [label1_per,label2_per,label3_per]
    #
    if min(label_per) > reject_threshold:
        decide = 0
    else:
        decide = label_per.index(min(label_per)) + 1
    # 0: Nonkeyword 1:알파봇 2:온누리 3:미리내 
    return decide

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='2018_spring_Speech Recognition system_final project_Keyword Spotting.')
    #PATH
    parser.add_argument('--data_path', default='./feature_saved', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--ex_name', default='noname', type=str)
    parser.add_argument('--continue_from', default=0, type=int)
    # Experiment
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cuda', default=2, type=int)

    parser.add_argument('--optimizer', default='SGD', type=str,choices=['SGD','Adam'], help='optimizer choose')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--reject_threshold', default=0.7, type=float, help='keyword reject_threshold')
    args = parser.parse_args()
    # 
    # 1. Load data and make data loader.
    assert args.batch_size == 1, 'Error: Batch size should be 1! If not the test_wrong_number will be wrong.'
    test_loader = data_load(path=args.data_path, is_training='Test', batch_size=args.batch_size)

    CSV_saver = CSV_saver(args.ex_name,args.continue_from)
    _, save_path = CSV_saver.get_path()
    # loss function: ctc loss.
    print("By mistake we didn't add blank label in the preprocessing(label.np generation) step.")
    labels = phn ##### By mistake we didn't add blank label in the preprocessing(label.np generation) step.
    decoder = GreedyDecoder(labels)

    #################### MODEL LOAD ####################

    if args.continue_from:
        print('*'*10 + 'continue from: ' + str(args.continue_from) + '*'*10)
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
        raise ValueError('shoud give integer to continue_from')

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    
    #################### MAKE TEST CSV ####################

    result_path_ = os.path.join('./results', args.ex_name)
    reject_threshold_result  = result_path_ + '/reject_threshold_result.csv'
    fd_reject_result = open(reject_threshold_result,'w')
    fd_reject_result.write('step,reject value,acc,number of wrong,ROC_TP,ROC_TN\n')

    #################### START TEST ####################
    def test(reject_threshold):
        model.eval()
        total_per=0
        test_acc = 0
        ROC_TP, ROC_TN = 0,0
        GT_T, GT_N = 0, 0
        for idx, data in enumerate(tqdm(test_loader)):
            inputs, targets, input_percentages,target_sizes = data
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

            label1 = 'a xl p a b o xd '
            label2 = 'o xn n u r i '
            label3 = 'm i r i n E '
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                per += decoder.wer(transcript, reference) / float(len(reference))
                final_decide = decide_label(transcript,reject_threshold)
                #print('final decide: ',final_decide)
                CSV_saver.write(args.continue_from,(transcript,reference),'test_result')
                if reference == label1:
                    if final_decide == 1:
                        test_acc += 1
                elif reference == label2:
                    if final_decide == 2:
                        test_acc += 1
                elif reference == label3:
                    if final_decide == 3:
                        test_acc += 1
                else:
                    if final_decide == 0:
                        test_acc += 1
                # Compute ROC
                if (reference==label1 or reference==label2 or reference==label3):
                    GT_T += 1
                    if (final_decide==1 or final_decide==2 or final_decide==3):
                        #print(reference,final_decide)
                        ROC_TP += 1
                else:
                    GT_N += 1
                    if final_decide == 0:
                        #print(reference,final_decide)
                        ROC_TN += 1
                # Done
            total_per += per

            if args.cuda:
                torch.cuda.synchronize()
            del out

        per = total_per / len(test_loader.dataset)
        per *= 100
        test_wrong_number = len(test_loader) - test_acc
        test_acc /=len(test_loader)
        test_acc *= 100
        per_results = per
        print('Test Average per {per:.3f}\tTest Accuracy:{acc:.3f}\tWrong Predicted Number{wn:}:'
            .format(per=per,acc=test_acc,wn=test_wrong_number))
        print('GT_T: {}, GT_N: {}'.format(GT_T,GT_N))
        return test_acc, test_wrong_number, ROC_TP, ROC_TN

    #################### ITERATE TEST ####################
    
    reject_threshold = 0.1
    FPR, TPR = [], []
    for i in range(20):
        test_acc, test_wrong_number,ROC_TP, ROC_TN = test(reject_threshold)
        print('For reject threshold [{}] result: acc={:.4f}\twrong_number={}'.format(reject_threshold,test_acc,test_wrong_number))
        fd_reject_result.write(str(start_epoch)+","+ str(reject_threshold) + ',' + str(test_acc) + ',' + str(test_wrong_number) + ',' + str(ROC_TP) + ',' + str(ROC_TN) +"\n")
        fd_reject_result.flush()
        reject_threshold += 0.1
    fd_reject_result.close()
    
    #################### NO ITERATE TEST ####################
    '''
    reject_threshold = 0.8
    _, _ = test(reject_threshold)
    '''