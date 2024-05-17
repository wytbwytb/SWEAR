from torch.autograd import Variable, Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_loss
from mmdet.models.losses import accuracy
from ...core.bbox.coder.angle_coder import SXYCoder
class IncompleteFeatSimulator(nn.Module):
    def __init__(self, init_channels, dim):
        super(IncompleteFeatSimulator, self).__init__()
        # self.conv_f1 = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=1, padding=0)
        # self.conv_f2 = nn.Conv1d(in_channels=3, out_channels=init_channels, kernel_size=1, padding=0)

        # self.conv1_1 = nn.Conv1d(init_channels, init_channels, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv1d(init_channels, init_channels, kernel_size=3, padding=1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.conv2_1 = nn.Conv1d(init_channels, init_channels * 2, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv1d(init_channels * 2, init_channels * 2, kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.conv_a1 = nn.Conv1d(in_channels=init_channels * 2, out_channels=1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)
        
        
        self.block1 = U_block(dim)
        self.block2 = U_block(dim)
        

    # def forward(self, x_feat, x_angle):
    def forward(self, x_feat, x_angle, y_angle):
        x_feat_out = x_feat.clone()

        _, levelx = torch.max(x_angle, dim=-1)
        _, levely = torch.max(y_angle, dim=-1)
        angle_max = torch.max(levelx, levely)

        # angle_max = torch.tensor(angle_max, dtype=torch.int64)
        # l1 = angle_max < 3
        
        # l1 = angle_max < 5
        # l2 = torch.logical_xor(l1, angle_max < 7)
        # l3 = angle_max > 6
        # _, level = torch.max(angle_max, dim=-1)
        l1 = angle_max == 0
        l2 = angle_max == 1
        l3 = angle_max == 2
        idx_l2 = torch.nonzero(l2).squeeze(dim=-1)
        idx_l3 = torch.nonzero(l3).squeeze(dim=-1)

        feats_l2 = torch.index_select(x_feat_out, 0, idx_l2)
        feats_l3 = torch.index_select(x_feat_out, 0, idx_l3)

        x_l3 = self.fc1(feats_l3)
        x_l3 = self.fc2(x_l3)
        x_l3 = self.fc3(x_l3)
        x_l3 = self.fc4(x_l3)

        # x_l3 = self.relu(self.bn(x_l3))

        x_l2 = self.fc3(feats_l2)
        x_l2 = self.fc4(x_l2)
        # x_l2 = self.relu(self.bn(x_l2))

        x_feat_out[idx_l2, :] = x_l2
        x_feat_out[idx_l3, :] = x_l3

        # x_feat_out = self.relu(self.bn(x_feat_out))

        # print(self.fc1.parameters().grad)
        return x_feat_out
        # x_full_feat = x_full_feat.repeat(proposal_size, 1)
        # x_full_feat = torch.unsqueeze(x_full_feat, dim=1)
        
        # x_angle = torch.unsqueeze(x_angle, dim=1)
        # x_angle = self.conv_f1(x_angle)
        # x_angle = x_angle.permute(0, 2, 1)
        # x_angle = torch.unsqueeze(x_angle, dim=1)
        # x_angle = x_angle.repeat(1, cls_num, 1, 1)
        # x_angle = x_angle.view(-1, 2, dim)


        # x = torch.cat([x_full_feat, x_angle], dim=1)
        # x = self.conv_f2(x)

        # x = self.conv1_1(x)
        # x = self.conv1_2(x)
        # x = self.pool1(self.relu1(x))

        # x = self.conv2_1(x)
        # x = self.conv2_2(x)
        # x = self.pool2(self.relu2(x))

        # x = self.conv_a1(x)
        # x = torch.squeeze(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.relu3(x)
        # x = x.view(proposal_size, cls_num, -1)
        
        # print(self.conv_f2.weight.grad)
        # return x

    # def forward(self, x_feat, x_angle, y_angle):
    #     x_feat_out = x_feat.clone()

    #     _, levelx = torch.max(x_angle, dim=-1)
    #     _, levely = torch.max(y_angle, dim=-1)
    #     angle_max = torch.max(levelx, levely)

    #     l1 = angle_max == 0
    #     l2 = angle_max == 1
    #     l3 = angle_max == 2
        
    #     idx_l2 = torch.nonzero(l2).squeeze(dim=-1)
    #     idx_l3 = torch.nonzero(l3).squeeze(dim=-1)

    #     feats_l2 = torch.index_select(x_feat_out, 0, idx_l2)
    #     feats_l3 = torch.index_select(x_feat_out, 0, idx_l3)

    #     x_l3 = self.block2(x_l3)
    #     x_l3 = self.block1(x_l3)

    #     # x_l3 = self.relu(self.bn(x_l3))

    #     x_l2 = self.block1(x_l2)
    #     # x_l2 = self.relu(self.bn(x_l2))

    #     x_feat_out[idx_l2, :] = x_l2
    #     x_feat_out[idx_l3, :] = x_l3

    #     # x_feat_out = self.relu(self.bn(x_feat_out))

    #     # print(self.fc1.parameters().grad)
    #     return x_feat_out
    
    def loss_simulate(self, feats_pos, gts_pos, feats_full):
        num = gts_pos.shape[0]
        all_sim = []
        for i in range(num):
            cls = gts_pos[i].item()
            sim = F.cosine_similarity(feats_pos[i], feats_full[cls], dim=-1).unsqueeze(dim=0)
            all_sim.append(1-sim)
        
        return torch.sum(torch.cat(all_sim)) / num

class AnglePerecption2(nn.Module):
    def __init__(self, init_channels, cfg, ):
        super(AnglePerecption2, self).__init__()
        self.share_fc = nn.Linear(in_features=256 * 49, out_features=1024)
        self.fc_x = nn.Linear(in_features=1024, out_features=1)
        self.fc_y = nn.Linear(in_features=1024, out_features=1)
        self.loss = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

    def forward(self, x):
        x = x.flatten(1)
        x = self.share_fc(x)
        x_pred = self.fc_x(x)
        y_pred = self.fc_y(x)
        return x_pred, y_pred

    def loss_angle(self, x_pred, y_pred, simulated, angles):
        idx = torch.nonzero(simulated).squeeze()
        x_pre_sim = torch.index_select(x_pred, 0, idx)
        y_pre_sim = torch.index_select(y_pred, 0, idx)
        # print(angle_pre_sim)
        # angles_pred = torch.index_select(angle_pred, 0, idx)
        angles_sim = torch.index_select(angles, 0, idx)

        num = angles_sim.shape[0]
        label_weights = torch.ones(num).cuda()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        loss_x = self.loss(
                    x_pre_sim,
                    x_pre_sim,
                    label_weights,
                    avg_factor=avg_factor)
        loss_y = self.loss(
                    y_pre_sim,
                    x_pre_sim,
                    label_weights,
                    avg_factor=avg_factor)


class AnglePerecption(nn.Module):
    def __init__(self, init_channels, cfg, ):
        super(AnglePerecption, self).__init__()
        # self.relu = nn.ReLU()

        # self.fc1_flatten = nn.Linear(cfg['in_channels'] * cfg['roi_feat_size'] * cfg['roi_feat_size'], cfg['fc_out_channels'])
        # self.fc2_flatten = nn.Linear(cfg['fc_out_channels'], cfg['fc_out_channels'])

        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=init_channels, kernel_size=1, padding=0)

        # self.conv1_1 = nn.Conv1d(init_channels, init_channels, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv1d(init_channels, init_channels, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.conv2_1 = nn.Conv1d(init_channels, init_channels * 2, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv1d(init_channels * 2, init_channels * 2, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.conv2 = nn.Conv1d(in_channels=init_channels * 2, out_channels=1, kernel_size=1, padding=0)
        # self.fc1 = nn.Linear(int(cfg['fc_out_channels'] / 4), 256)
        self.fc_angle = nn.Linear(in_features=1024, out_features=3)
        # self.fc_x = nn.Linear(in_features=1024, out_features=3*11)
        # self.fc_y = nn.Linear(in_features=1024, out_features=3*11)
        self.fc_x = nn.Linear(in_features=1024, out_features=3)
        self.fc_y = nn.Linear(in_features=1024, out_features=3)
        # self.xy_coder = SXYCoder('triangle', 40)
        self.loss = build_loss(dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True, 
                     loss_weight=0.1))
        # self.loss = build_loss(dict(
        #              type='SmoothFocalLoss',
        #              gamma=2.0,
        #              alpha=0.25,
        #             #  use_sigmoid=True, 
        #              loss_weight=0.1))

    
    def forward(self, x):
        # x = x.flatten(1)
        # x = self.relu(self.fc1_flatten(x))
        # x = self.relu(self.fc2_flatten(x))

        # x = torch.unsqueeze(x, dim = 1)
        # x = self.conv1(x)

        # x = self.conv1_1(x)
        # x = self.conv1_2(x)
        # x = self.pool1(self.relu(x))

        # x = self.conv2_1(x)
        # x = self.conv2_2(x)
        # x = self.pool2(self.relu(x))

        # x = self.conv2(x)
        # x = torch.squeeze(x)
        # x = self.relu(self.fc1(x))

        x_pred = self.fc_x(x)
        y_pred = self.fc_y(x)
        # angle_pred = self.fc_angle(x)

        # print(self.fc_angle.weight.grad)
        # return angle_pred
        return x_pred, y_pred
    
    def loss_angle_onlysim(self, x_pre_sim, y_pre_sim, angles_sim):
        # angles_sim = torch.cat(angles_sim)
        angles_sim_l1 =torch.floor(angles_sim / 50).long()
        angles_sim_l2 =torch.floor(angles_sim / 70).long()
        l2_idx = angles_sim_l2 >= 1
        angles_sim_l1[l2_idx] = 2.0
        x_label_sim = angles_sim_l1[:, 0]
        y_label_sim = angles_sim_l1[:, 1]
        
        num = angles_sim.shape[0]
        label_weights = torch.ones(num).cuda()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        loss_x = self.loss(
                    x_pre_sim,
                    x_label_sim,
                    label_weights,
                    avg_factor=avg_factor)
        loss_y = self.loss(
                    y_pre_sim,
                    y_label_sim,
                    label_weights,
                    avg_factor=avg_factor)

        return (loss_x + loss_y) / 2
    
    def loss_angle(self, x_pred, y_pred, simulated, angles, gts, scores):
        # bg_class_ind = 10
        # pos_inds = (labels >= 0) & (labels < bg_class_ind)
        _, top_label = torch.max(scores, dim=-1)
        sim_inds = (simulated > 0)
        # sim_inds = (simulated > 0) & (top_label < 10)
        
        
        x_pre_sim = x_pred[sim_inds.type(torch.bool)]
        y_pre_sim = y_pred[sim_inds.type(torch.bool)]
        # x_pre_sim = x_pre_sim[idx2]

        # x_pre_sim = x_pred.view(x_pred.size(0), -1, 3)[sim_inds.type(torch.bool),gts[sim_inds.type(torch.bool)]]
        # y_pre_sim = y_pred.view(y_pred.size(0), -1, 3)[sim_inds.type(torch.bool),gts[sim_inds.type(torch.bool)]]
       
       
       
        # idx = torch.nonzero(simulated).squeeze()
        # x_pre_sim = torch.index_select(x_pred, 0, idx)
        # y_pre_sim = torch.index_select(y_pred, 0, idx)
        # print(angle_pre_sim)
        # angles_pred = torch.index_select(angle_pred, 0, idx)

        angles_sim = angles[sim_inds.type(torch.bool)]

        # x_sim = angles_sim[:, 0]
        # y_sim = angles_sim[:, 1]

        # angle_sim = (x_sim + y_sim) / 2
        # angle_sim = torch.max(x_sim, y_sim)
        # angle_label_sim = self.xy_coder.encode(angle_sim.unsqueeze(dim=-1))

        # x_label_sim = self.xy_coder.encode(x_sim.unsqueeze(dim=-1))
        # y_label_sim = self.xy_coder.encode(y_sim.unsqueeze(dim=-1))

        # # make labels
        # angles_sim_l1 =torch.floor(angle_sim / 50).long()
        # angles_sim_l2 =torch.floor(angle_sim / 70).long()
        # l2_idx = angles_sim_l2 >= 1
        # angles_sim_l1[l2_idx] = 2.0
        # angle_label_sim = angles_sim_l1

        angles_sim_l1 =torch.floor(angles_sim / 50).long()
        angles_sim_l2 =torch.floor(angles_sim / 70).long()
        l2_idx = angles_sim_l2 >= 1
        angles_sim_l1[l2_idx] = 2.0
        x_label_sim = angles_sim_l1[:, 0]
        y_label_sim = angles_sim_l1[:, 1]

        # x_label_sim = torch.floor(angles_sim[:, 0] / 10).long()
        # y_label_sim = torch.floor(angles_sim[:, 1] / 10).long()

        num = angles_sim.shape[0]
        label_weights = torch.ones(num).cuda()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        loss_x = self.loss(
                    x_pre_sim,
                    x_label_sim,
                    label_weights,
                    avg_factor=avg_factor)
        loss_y = self.loss(
                    y_pre_sim,
                    y_label_sim,
                    label_weights,
                    avg_factor=avg_factor)

        # loss = self.loss(
        #     angles_pred,
        #     angle_label_sim,
        #     label_weights,
        #     avg_factor=avg_factor
        # )

        # x_acc = accuracy(x_pre_sim, x_label_sim)
        # y_acc = accuracy(y_pre_sim, y_label_sim)
        # print('x_acc:', x_acc, 'y_acc:', y_acc)

        # loss_con = torch.tensor(0.).cuda()
        # for i in range(11 - 1):
        #     idx = torch.fmod(gts[sim_inds.type(torch.bool)] + i + 1, 11)
        #     x_pre_labeli = x_pred.view(x_pred.size(0), -1, 3)[sim_inds.type(torch.bool),idx]
        #     y_pre_labeli = y_pred.view(y_pred.size(0), -1, 3)[sim_inds.type(torch.bool),idx]
        #     loss_xi = self.loss(
        #         x_pre_labeli,
        #         x_label_sim,
        #         label_weights,
        #         avg_factor=avg_factor)
        #     loss_yi = self.loss(
        #         y_pre_labeli,
        #         y_label_sim,
        #         label_weights,
        #         avg_factor=avg_factor)
        #     # loss_con += ((1 - loss_xi) + (1 - loss_yi))/2
        #     loss_con_x = torch.max(loss_xi - loss_x, torch.tensor(0.).cuda()) 
        #     loss_con_y = torch.max(loss_yi - loss_y, torch.tensor(0.).cuda()) 
        #     loss_con += loss_con_x + loss_con_y
        # loss_con = loss_con / (11 - 1)

        return (loss_x + loss_y) / 2

    # def loss_angle(self, x_pred, y_pred, simulated, angles):
    # # def loss_angle(self, angle_pred, simulated, angles):
    #     idx = torch.nonzero(simulated).squeeze()
    #     x_pre_sim = torch.index_select(x_pred, 0, idx)
    #     y_pre_sim = torch.index_select(y_pred, 0, idx)
    #     # print(angle_pre_sim)
    #     # angles_pred = torch.index_select(angle_pred, 0, idx)
    #     angles_sim = torch.index_select(angles, 0, idx)

    #     # x_sim = angles_sim[:, 0]
    #     # y_sim = angles_sim[:, 1]

    #     # angle_sim = (x_sim + y_sim) / 2
    #     # angle_sim = torch.max(x_sim, y_sim)
    #     # angle_label_sim = self.xy_coder.encode(angle_sim.unsqueeze(dim=-1))

    #     # x_label_sim = self.xy_coder.encode(x_sim.unsqueeze(dim=-1))
    #     # y_label_sim = self.xy_coder.encode(y_sim.unsqueeze(dim=-1))

    #     # # make labels
    #     # angles_sim_l1 =torch.floor(angle_sim / 50).long()
    #     # angles_sim_l2 =torch.floor(angle_sim / 70).long()
    #     # l2_idx = angles_sim_l2 >= 1
    #     # angles_sim_l1[l2_idx] = 2.0
    #     # angle_label_sim = angles_sim_l1

    #     # angles_sim_l1 =torch.floor(angles_sim / 50).long()
    #     # angles_sim_l2 =torch.floor(angles_sim / 70).long()
    #     # l2_idx = angles_sim_l2 >= 1
    #     # angles_sim_l1[l2_idx] = 2.0
    #     # x_label_sim = angles_sim_l1[:, 0]
    #     # y_label_sim = angles_sim_l1[:, 1]

    #     x_label_sim = torch.floor(angles_sim[:, 0] / 10).long()
    #     y_label_sim = torch.floor(angles_sim[:, 1] / 10).long()

    #     num = angles_sim.shape[0]
    #     label_weights = torch.ones(num).cuda()
    #     avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

    #     loss_x = self.loss(
    #                 x_pre_sim,
    #                 x_label_sim,
    #                 label_weights,
    #                 avg_factor=avg_factor)
    #     loss_y = self.loss(
    #                 y_pre_sim,
    #                 y_label_sim,
    #                 label_weights,
    #                 avg_factor=avg_factor)

    #     # loss = self.loss(
    #     #     angles_pred,
    #     #     angle_label_sim,
    #     #     label_weights,
    #     #     avg_factor=avg_factor
    #     # )

    #     # x_acc = accuracy(x_pre_sim, x_label_sim)
    #     # y_acc = accuracy(y_pre_sim, y_label_sim)
    #     # print('x_acc:', x_acc, 'y_acc:', y_acc)
    #     return (loss_x + loss_y) / 2

        # return loss

class U_block(nn.Module):
    def __init__(self, dim):
        super(U_block, self).__init__()
        self.down_linear1 = nn.Linear(dim, int(dim / 2))
        self.down_linear2 = nn.Linear(int(dim / 2), int(dim / 4))
        self.up_linear1 = nn.Linear(int(dim / 2), int(dim / 4))
        self.up_linear1 = nn.Linear(int(dim / 2), int(dim / 4))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.down_linear1(x)
        x = self.relu(self.down_linear2(x))
        x = self.up_linear1(x)
        x = self.up_linear2(x)
        return x

def cal_loss_angle(angle_pre, simulated, angles):
    # reg
    idx = torch.nonzero(simulated).squeeze()
    angle_pre_sim = torch.index_select(angle_pre, 0, idx)
    # print(angle_pre_sim)
    angles_sim = torch.index_select(angles, 0, idx)
    dx_angles = angles_sim[:, 0:2] / 90
    loss = F.smooth_l1_loss(angle_pre_sim, dx_angles)

    # cls
    # loss_func = build_loss(dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=False,
    #                  loss_weight=1.0))
    return loss

def cal_simi(feats_sim, feats_full):
    cls_num = feats_full.shape[0]
    # feats_sim = feats_sim.permute(1, 0, 2)
    cls_sim = []
    # feats = feats[:20, :]
    # feats_sim = feats_sim[:, :20, :]
    for i in range(cls_num):
        sim = F.cosine_similarity(feats_sim, feats_full[i], dim=-1).unsqueeze(dim=-1)
        cls_sim.append(sim)
    all_sim = torch.cat(cls_sim, dim=1)
    return all_sim
