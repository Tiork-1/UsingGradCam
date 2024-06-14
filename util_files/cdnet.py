from __future__ import print_function
from torch.autograd import Variable
from torch.optim import Adam
from util_files.transferLearning_clfHeads import softMax, cosMax, arcMax
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from util_files.utils import device_kwargs
from util_files.weightnet import WeightNet
from util_files.discriminator import Discriminator
from util_files.utils2 import generate_flip_grid, ACLoss
import torch.nn.functional as F

def clf_fun(self, n_class, device, s=20, m=0.01):
    if self.method == 'softMax':
        clf = softMax(self.out_dim, n_class).to(device)
    elif self.method == 'cosMax':
        clf = cosMax(self.out_dim, n_class, s).to(device)
    elif self.method == 'arcMax':
        clf = arcMax(self.out_dim, n_class, s, m).to(device)

    return clf

class CDNet(nn.Module):
    def __init__(self, args, net, n_class):
        super(CDNet, self).__init__()

        self.device = device_kwargs(args)
        self.method = args.method
        self.lr = args.lr
        self.backbone = args.backbone
        self.args = args

        self.n_epoch = args.n_epoch
        self.n_class = n_class
        self.n_support = args.n_shot
        self.n_query = args.n_query

        self.out_dim = args.out_dim
        self.lr = args.lr

        # encoder
        self.net = net.to(self.device)
        self.over_fineTune = args.over_fineTune

        # decomposition
        self.colors = args.color
        self.feat_dim = 512

        self.decomposition = nn.Sequential(
            nn.Linear(self.out_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.PReLU()
        )

        # weightnet
        self.weight_net = WeightNet(self.out_dim + self.feat_dim)

        # discriminator
        self.discriminator = Discriminator(self.feat_dim, args.n_domains)

        # classifier
        self.base_clf = clf_fun(self, self.n_class, self.device)

        self.ft_n_epoch = args.ft_n_epoch
        self.n_way = args.n_way

        # fix encoder
        self.optimizer = Adam([{'params': self.decomposition.parameters()},
                               {'params': self.discriminator.parameters()},
                               {'params': self.weight_net.parameters()},
                               {'params': self.base_clf.parameters()}],
                              lr=1e-4)

    def accuracy_fun_tl(self, data_loader):
        Acc = 0
        self.net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = Variable(x).to(self.device), Variable(y).to(self.device)
                logits = self.clf(self.net(x))
                y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
                Acc += np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item() / len(data_loader)

    def acc_feature(self, logits, y):
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
        Acc = np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item()
    def accuracy_fun(self, x, n_way):
        x_support = x[:, :self.n_support, :, :, :].contiguous()
        x_support = x_support.view(n_way * self.n_support, *x.size()[2:])

        x_query = x[:, self.n_support:, :, :, :].contiguous()
        x_query = x_query.view(n_way * self.n_query, *x.size()[2:])
        y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        with torch.no_grad():
            z_support, _, _, _, _, _ = self.extract_feature(x_support)
            z_query, _, _, _, _, _ = self.extract_feature(x_query)

        # proto classifier
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)

        dists = self.euclidean_dist(z_query, z_proto)
        scores = -dists

        y_hat = np.argmax(scores.data.cpu().numpy(), axis=1)

        return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int)) * 100

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


    def extract_feature(self, x):
        x, feat_map = self.net(x)  # ori feature
        ori_x = x

        re_x = 0.0

        primarys = []
        weights = []

        for i in range(self.colors):
            # decompostion
            primary = self.decomposition(x)
            # primary = primary.mean(dim=0)  # 取当前batch的均值
            input_f = torch.cat([x, primary.expand(x.shape[0], self.out_dim)], dim=1)
            weight = self.weight_net(input_f)

            primarys.append(primary)
            weights.append(weight)

            x = x - primary * weight

            re_x += primary * weight

        primarys = torch.stack(primarys)  # [color, b, dim]
        weights = torch.stack(weights)  # [color,batch,1]

        return re_x, x, ori_x, primarys, weights, feat_map

    def forward(self, x):
        self.net.train()
        self.decomposition.train()
        self.weight_net.train()
        self.base_clf.train()
        self.discriminator.train()
        loss_sum = 0

        x = Variable(x).to(self.device)
        
        # forward
        re_x, de_x, ori_x, primarys, weights, feat_map = self.extract_feature(x)

        output = self.base_clf(de_x)

        return de_x
    
    
    # note: this is typical/batch based training
    def train_loop(self, x):
        self.net.train()
        self.decomposition.train()
        self.weight_net.train()
        self.base_clf.train()
        self.discriminator.train()
        loss_sum = 0

        x = Variable(x).to(self.device)
        
        # forward
        re_x, de_x, ori_x, primarys, weights, feat_map = self.extract_feature(x)

        output = self.base_clf(de_x)

        return de_x

    def test_fer(self, testLoader):
        self.net.eval()

        correct = 0
        total = 0

        for i, (x, y) in enumerate(testLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            scores = self.base_clf(self.net(x))
            pred = scores.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += len(y)

        return (correct / total) * 100.0

    # note this is episodic testing
    def test_loop(self, test_loader, n_way, epoch):
        acc_all = []
        self.net.eval()
        self.decomposition.eval()
        self.weight_net.eval()
        self.discriminator.eval()
        for i, (x, _) in enumerate(test_loader):
            x = Variable(x).to(self.device)
            self.n_query = x.size(1) - self.n_support
            acc = self.accuracy_fun(x, n_way)
            acc_all.append(acc)

            if i % 10 == 0:
                print("avg acc of task {}: {}".format(i, np.mean(acc_all)))

        acc_all = np.asarray(acc_all)
        teAcc = np.mean(acc_all)  # n_episodes 的均值
        acc_std = np.std(acc_all)
        conf_interval = 1.96 * acc_std / np.sqrt(len(test_loader))

        return teAcc, conf_interval
