#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:54:02 2021

@author: fatemeh tahrirchi

"""

import datasets,net
from preprocessing import Preprocessing,CharVectorizer
from net import VDCNN,train,save
import lmdb
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os, subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MODELS_FOLDER = 'models/vdcnn'
DATA_FOLDER = 'datasets'
DATASET='yelp_review_full'#['yelp_review_full','yelp_review_polarity']
PREPROCES_TYPE='lower'#['lower','denoiser','add_pos','add_hashtag','add_NOT']

# get device to calculate on (either CPU or GPU with minimum memory load)
def get_gpu_memory_map():
    
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    
    return gpu_memory_map

def get_device():
    if torch.cuda.is_available():
        memory_map = get_gpu_memory_map()
        device = "cuda:%d" % min(memory_map, key=memory_map.get)
    else:
        device = "cpu"
    
    print("Device:", device)
    return device



class TupleLoader(Dataset): #torch.utils.dat.Dataset

    def __init__(self, path=""):
        self.path = path

        self.env = lmdb.open(path, max_readers=opt.nthreads, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        return xtxt, lab

def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)


#-------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET, help="'yelp_review_full' or 'yelp_review_polarity'")
    parser.add_argument("--preproces_type", type=str, default=PREPROCES_TYPE, help="'lower' or 'denoiser' or 'add_pos' or 'add_hashtag' or 'add_NOT'")
    parser.add_argument("--model_folder", type=str, default=MODELS_FOLDER+"/"+DATASET, help="result directory")
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER+"/"+DATASET, help="address of datasets directory")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=29, help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=1024, help="max lentgh of input string")
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--solver", type=str, default="sgd", help="'sgd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=10, help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--nthreads", type=int, default=4)
   
    args,_ = parser.parse_known_args()
    return args


#---Main-------


opt = get_args()
print("parameters: {}".format(vars(opt)))
    
os.makedirs(opt.model_folder, exist_ok=True)
os.makedirs(opt.data_folder, exist_ok=True)

dataset = datasets.load_datasets(opt.dataset)
dataset_name = dataset.data_name
n_classes = dataset.n_classes
print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

tr_path =  "{}/train.lmdb".format(opt.data_folder)
te_path = "{}/test.lmdb".format(opt.data_folder)
    
# check if datasets exis
all_exist = True if (os.path.exists(tr_path) and os.path.exists(te_path)) else False
all_exist=False
preprocessor = Preprocessing(opt.preproces_type)
vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='pre')
n_tokens = len(vectorizer.char_dict)

if not all_exist:
    print("Creating datasets")
    tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
    te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
            
    n_tr_samples = len(tr_sentences)
    n_te_samples = len(te_sentences)
    del tr_sentences
    del te_sentences

    print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

    ###################
    # transform train #
    ###################
    with lmdb.open(tr_path, map_size=1099511627776) as env:
        with env.begin(write=True) as txn:
            for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

                xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                lab = label

                txt_key = 'txt-%09d' % i
                lab_key = 'lab-%09d' % i
                    
                txn.put(lab_key.encode(), list_to_bytes([lab]))
                txn.put(txt_key.encode(), list_to_bytes(xtxt))

            txn.put('nsamples'.encode(), list_to_bytes([i+1]))

    ##################
    # transform test #
    ##################
    with lmdb.open(te_path, map_size=1099511627776) as env:  #
        with env.begin(write=True) as txn:
            for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):

                xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                lab = label

                txt_key = 'txt-%09d' % i
                lab_key = 'lab-%09d' % i
                    
                txn.put(lab_key.encode(), list_to_bytes([lab]))
                txn.put(txt_key.encode(), list_to_bytes(xtxt))

            txn.put('nsamples'.encode(), list_to_bytes([i+1]))

                
tr_loader = DataLoader(TupleLoader(tr_path), batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
te_loader = DataLoader(TupleLoader(te_path), batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=False) #num_workers=opt.nthreads

# select cpu or gpu
device = get_device()
list_metrics = ['accuracy']


print("Creating model...")
net = VDCNN(n_classes=n_classes, num_embedding=int(n_tokens + 1), embedding_dim=16, depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)
criterion = torch.nn.CrossEntropyLoss()
net.to(device)

assert opt.solver in ['sgd', 'adam']
if opt.solver == 'sgd':
    print(" - optimizer: sgd")
    optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr, momentum=opt.momentum)
elif opt.solver == 'adam':
    print(" - optimizer: adam")
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr)    
        
scheduler = None
if opt.lr_halve_interval and  opt.lr_halve_interval > 0:
    print(" - lr scheduler: {}".format(opt.lr_halve_interval))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
        
for epoch in range(1, opt.epochs + 1):
    train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion,list_metrics=list_metrics)
    train(epoch,net, te_loader, device, msg="testing ", criterion=criterion,list_metrics=list_metrics)

    if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
        path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
        print("snapshot of model saved as {}".format(path))
        save(net, path=path)


if opt.epochs > 0:
    path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
    print("snapshot of model saved as {}".format(path))
    save(net, path=path)
