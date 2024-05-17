from torch.autograd import Variable,Function
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(Function):

    @ staticmethod
    def forward(self, x):
        return x.view_as(x)

    @ staticmethod
    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -1.0)


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x)


def make_classify(classifier, proposals_feat, proposals_domain):
    #gpdc = gpDomainClassifier(sgp.shape[1] * sgp.shape[0], sgp.shape[0]).cuda()
    #type(sgp.cuda())
    #pdb.set_trace()
    #.view(proposals_feat.shape[0],1)
    # scores = classifier(grad_reverse(proposals_feat))
    scores = classifier(proposals_feat)
    scores = scores.squeeze(-1)
    #pdb.set_trace()    
    loss_domain = F.binary_cross_entropy(scores, proposals_domain.float())

    return loss_domain


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator,self).__init__()
        self.x1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.x2 = nn.LeakyReLU(0.2)
        self.x3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.x4 = nn.BatchNorm1d(512)

        self.cls_pred = nn.Linear(512, 10)
        
        self.y1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.y2 = nn.LeakyReLU(0.2)
        self.y3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.y4 = nn.BatchNorm1d(256)

        self.z1 = nn.Linear(256,1)
        # self.z2 = nn.Sigmoid()
        self.z2 = nn.Tanh()

        
        # self.dis = nn.Sequential(
        #     nn.Linear(dim,512),
        #     # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     # nn.AvgPool1d(kernel_size=2, stride=2),
        #     nn.BatchNorm1d(512),
 
        #     nn.Linear(512,256),
        #     # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     # nn.AvgPool1d(kernel_size=2, stride=2),
        #     nn.BatchNorm1d(256),
 
        #     # nn.Linear(256,128),
        #     # nn.LeakyReLU(0.2),
        #     # nn.BatchNorm1d(128),
 
        #     nn.Linear(256,1),
        #     nn.Sigmoid()
        # )
 
    def forward(self,x):
        x = self.x1(x.view(-1, 1, x.shape[-1]))
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x.view(-1, x.shape[-1]))

        cls_pred = self.cls_pred(x)

        x = self.y1(x.view(-1, 1, x.shape[-1]))
        x = self.y2(x)
        x = self.y3(x)
        x = self.y4(x.view(-1, x.shape[-1]))

        x = self.z1(x)
        x = torch.sigmoid(self.z2(x))

        # x=self.dis(x)
        return x, cls_pred


class DomainClassifier(nn.Module):
    def __init__(self, dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = x.view(x.size(0), -1)
        return x


def loss_D(net, pos_feats_flatten, pos_simulated, pos_gts):
    criteria = torch.nn.BCEWithLogitsLoss()
    criteria_cls = torch.nn.CrossEntropyLoss()
    sim_idx = pos_simulated > 0
    # sim = pos_simulated[sim_idx]
    real_idx = pos_simulated == 0
    # real = pos_simulated[real_idx]
    # pos_feats_flatten = pos_feats_flatten.unsqueeze(dim=1)
    pred, cls_pred = net(pos_feats_flatten)
    sim_pred = pred[sim_idx]
    real_pred = pred[real_idx]
    target_sim = torch.zeros_like(sim_pred, requires_grad=False)
    target_real = torch.ones_like(real_pred, requires_grad=False)
    loss_sim = criteria(sim_pred, target_sim)
    loss_real = criteria(real_pred, target_real)
    loss_cls = criteria_cls(cls_pred, pos_gts)
    
    return loss_sim + loss_real + loss_cls

def loss_G(net, pos_feats_flatten, pos_simulated, pos_gts):
    criteria = torch.nn.BCELoss()
    pred, cls = net(pos_feats_flatten)
    target = torch.ones_like(pred, requires_grad=False)
    loss_g = criteria(pred, target)
    return loss_g

    

def freeze_params(net):
    for param in net.parameters():
        param.requires_grad = False

def unfreeze_params(net):
    for param in net.parameters():
        param.requires_grad = True
