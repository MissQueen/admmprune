import argparse

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.nn as nn
from CusOptim import CusAdam
# 全局取消证书验证
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def regularized_crossentropy(args, model, loss, output, target):
    curloss = loss(output, target)
    index = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)) :
            curloss += args.lamda * (param.norm())
            index += 1
    return curloss

def test(args, model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader), desc='Loop:Test',
                                      total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        correct /= len(test_loader.dataset)
        print('test_loss:', test_loss,'correction:', correct)

def initialize_Z_U(model):
    Z = []
    U = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)):
            Z.append(param.detach().cpu().clone())
            U.append(torch.zeros_like(param).cpu())
    return Z,U

def admm_loss(args, model, loss, output, target, U, Z):
    curloss = loss(output, target)
    index = 0
    # print('U:',len(U),'Z:',len(Z))
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)):
            curloss += args.lamda * (param.norm())
            u = U[index].to(device)
            z = Z[index].to(device)
            curloss += (args.rho / 2) * ((param-z+u).norm())
            index += 1
    return curloss

def update_W(model):
    W = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)):
            W.append(param.detach().cpu().clone())
    return W


def update_Z(W, U, args):
    Z = []
    index = 0
    for w, u in zip(W, U):
        z = w + u
        pcen = np.percentile(abs(z), 100*args.percent[index])
        under_threshold = abs(z)<pcen
        z.data[under_threshold] = 0
        Z.append(z)
        index += 1
    return Z

def update_U(W, Z, U):
    new_U = []
    for w,z,u in zip(W,Z,U):
        u = u + w - z
        new_U.append(u)

    return new_U

def prune_weight(param, device, percent):
    weight_numpy = param.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100 * percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask

def compute_weight_zeros(model):
    zeros = 0
    sum_params = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)):
            temp = torch.zeros_like(param)
            # print(param.shape, temp.shape)
            temp[param != 0] = 1
            sum_params += param.numel()
            zeros += (param.numel() - torch.sum(temp).cpu().detach().numpy())
    return zeros, sum_params

def apply_prune(model, device, args):
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (('conv' in name) or ('fc' in name)):
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def train(args, model, device, train_loader, test_loader, loss, optimizer):
    #==============pretrain==================
    epoch = 0
    while(True):
        model.train()
        sum_loss = 0
        for i, (data, target) in tqdm(enumerate(train_loader), desc=f'Epoch:{epoch}.Loop:Train',total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            curloss = regularized_crossentropy(args, model, loss, output, target)
            sum_loss += curloss.item()
            curloss.backward()
            optimizer.step()
        epoch+=1
        sum_loss /= len(train_loader.dataset)
        print('cur_loss:',sum_loss)
        test(args, model, device, test_loader, loss)
        if(sum_loss<0.0013):
            break
    torch.save(model.state_dict(), os.path.join(args.checkpoints, 'pre_depthnet.pth'))
    zeros, sum_params = compute_weight_zeros(model)
    print('pretrain zeros:',zeros, 'sum_params:', sum_params)
    #===============admm=====================

    Z, U = initialize_Z_U(model)

    for epoch in range(args.admm_epochs):
        model.train()
        for i, (data, target) in tqdm(enumerate(train_loader), desc=f'Epoch:{epoch}/{args.admm_epochs}.Loop:Train',total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            curloss = admm_loss(args, model, loss, output, target, U, Z)
            curloss.backward()
            optimizer.step()
        W = update_W(model)
        Z = update_Z(W, U, args)
        U = update_U(W,Z,U)
        test(args, model, device, test_loader, loss)
    zeros, sum_params = compute_weight_zeros(model)
    print('admm zeros:', zeros, 'sum_params:', sum_params)
    mask = apply_prune(model, device,args)
    zeros, sum_params = compute_weight_zeros(model)
    print('emmm zeros:', zeros, 'sum_params:', sum_params)
    torch.save(model.state_dict(), os.path.join(args.checkpoints, 'admm_depthnet.pth'))
    #=================retrain=================

    for epoch in range(args.pruned_epochs):
        model.train()
        for i, (data, target) in tqdm(enumerate(train_loader), desc=f'Epoch:{epoch}/{args.pruned_epochs}.Loop:Train',total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            curloss = regularized_crossentropy(args,model,loss,output, target)
            curloss.backward()
            optimizer.prune_step(mask)
        test(args, model, device, test_loader, loss)
    torch.save(model.state_dict(), os.path.join(args.checkpoints, 'pruned_depthnet.pth'))
    zeros, sum_params = compute_weight_zeros(model)
    print('pruned zeros:', zeros, 'sum_params:', sum_params)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoints',type=str, default='/disk1/lcx/cifa')
    parser.add_argument('--percent', type=list, default=[0.8,0.92,0.991,0.93,0.91,0.92,0.93,0.94,0.92,0.93,
                                                         0.88,0.92,0.991,0.93,0.91,0.92,0.93,0.94,0.92,0.93,
                                                         0.89,0.92,0.991,0.93,0.91,0.92,0.93,0.94,0.92,0.93,
                                                         0.90,0.92,0.991,0.93,0.91,0.92,0.93,0.94,0.92,0.93,
                                                         0.89,0.92,0.991,0.93,0.91,0.92,0.93,0.94,0.92,0.93])
    parser.add_argument('--lamda', type=float, default=0.00005)
    parser.add_argument('--pre_epochs', type=int, default=10)
    parser.add_argument('--admm_epochs', type=int, default=20)
    parser.add_argument('--pruned_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--rho',type=float, default=1e-2)

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('device:',device)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.checkpoints, train=True, download=True,
                         transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                  (0.24703233, 0.24348505, 0.26158768))
                         ])),shuffle=True, batch_size=args.batch_size)
    print('train_loader load...')
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.checkpoints, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                  (0.24703233, 0.24348505, 0.26158768))
                         ])), shuffle=True, batch_size=1000)
    print('test_loader load...')

    model = models.resnet50(num_classes=10, pretrained=False).to(device)
    optimizer = CusAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    loss = nn.CrossEntropyLoss()
    train(args, model, device, train_loader, test_loader, loss, optimizer)

