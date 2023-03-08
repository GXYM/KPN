import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VggNet
from network.resnet import ResNet
# from util.config import config as self.cfg, update_config, print_config
from util.option import BaseOptions
from util.misc import mkdirs, to_device
import time
import numpy as np
import cv2

####################### KPN head ####################################
class Feature_head(nn.Module):
    def __init__(self, in_channels, mid_channel=8, cls_channel=1, centergauss_channel=1, embedding_channel=32, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(mid_channel),
            nn.PReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(mid_channel),
            nn.PReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(mid_channel),
            nn.PReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(mid_channel),
            nn.PReLU(),
            nn.Conv2d(mid_channel, cls_channel, kernel_size=3, stride=1, padding=1),
        )
        if self.cfg.add_focal_loss:
            self.conv_center = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                # nn.Conv2d(mid_channel, centergauss_channel, kernel_size=3, stride=1, padding=1),
            )
            self.conv_center_head = nn.Conv2d(mid_channel, centergauss_channel, kernel_size=3, stride=1, padding=1)
            nn.init.constant_(self.conv_center_head.bias, -2.19)
        else:
            self.conv_center = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, centergauss_channel, kernel_size=3, stride=1, padding=1),
            )

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels+2, embedding_channel, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(num_channels=embedding_channel, num_groups=2),
            nn.BatchNorm2d(embedding_channel),
            nn.PReLU(),
            nn.Conv2d(embedding_channel, embedding_channel, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(num_channels=embedding_channel, num_groups=2),
            nn.BatchNorm2d(embedding_channel),
            nn.PReLU(),
            nn.Conv2d(embedding_channel, embedding_channel, kernel_size=3, stride=1, padding=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(embedding_channel, embedding_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(embedding_channel),
            nn.PReLU(),
            nn.Conv2d(embedding_channel, embedding_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(embedding_channel),
            nn.PReLU(),
            nn.Conv2d(embedding_channel, embedding_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, feat_in):
        n, c, h, w = feat_in.shape
        e_key = str(n)+"_"+str(h)+"_"+str(w)
        if e_key not in self.cfg.torch_xy_embedding:
            x_range = torch.linspace(-1, 1, feat_in.shape[-1], device=feat_in.device)
            y_range = torch.linspace(-1, 1, feat_in.shape[-2], device=feat_in.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([feat_in.shape[0], 1, -1, -1])
            x = x.expand([feat_in.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            self.cfg.torch_xy_embedding[e_key] = coord_feat

        torch_xy_embedding = self.cfg.torch_xy_embedding[e_key]
        # print("feat_in", feat_in.shape, torch_xy_embedding.shape)
        f_cls = None
        if self.cfg.cls_branch:
            f_cls = self.conv_cls(feat_in)
        f_centergauss = self.conv_center(feat_in)
        if self.cfg.add_focal_loss:
            f_centergauss = self.conv_center_head(f_centergauss)
        # f_centergauss = torch.sigmoid(self.conv_center(x))
        # f_centergauss = f_centergauss * torch.sigmoid(f_cls)

        feat_xy = torch.cat([feat_in, torch_xy_embedding], dim=1)
        feat_xy = self.conv_embedding(feat_xy)

        f_kernel = self.conv(feat_xy)
        f_kernel = torch.tanh(f_kernel)
        # f_kernel = f_kernel / (f_kernel.norm(dim=1, keepdim=True) + 0.0000009);
        # f_kernel /= torch.sqrt(torch.sqrt(torch.tensor(f_kernel.shape[1]).float()))
        #f_embedding = self.conv(feat_xy)

        return f_cls, f_centergauss, f_kernel, f_kernel
####################### END KPN head ####################################



#https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py
def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

####################### FPN ####################################

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.SepareConv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, upsampled, shortcut, up=True):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = self.BN1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = self.BN2(x)
        x = F.relu(x)
        if up:
            x = self.deconv(x)
        return x

class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', cfg=None, pretrain=False):
        super().__init__()
        self.backbone_name = backbone
        self.scale = cfg.scale

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=pretrain)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=pretrain)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=pretrain)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=pretrain)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            self.merge2 = UpBlok(256 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)
            self.merge1_scale2 = UpBlok(64 + 32, 32)
        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        if self.scale == 2:
            up2_1 = self.merge1_scale2(C1, up2, up=False)
            return up2_1, up3, up4, up5
        elif self.scale == 1:
            up1 = self.merge1(C1, up2)
            return up1, up2, up3, up4, up5
####################### END FPN ####################################




class KPN_Net(nn.Module):

    def __init__(self, backbone='vgg', is_training=True, cfg=None):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, cfg=cfg, pretrain=is_training)
        if cfg.scale == 1:
            self.feature = Feature_head(16, cfg=cfg)
        elif cfg.scale == 2:
            self.feature = Feature_head(32, cfg=cfg)
        self.cfg = cfg

        if self.cfg.add_focal_loss:
            self.KPN_focal_bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.KPN_focal_bias.data.fill_(-2.19)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        if self.cfg.cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict['model'])

    ####################### KPN module ####################################
    def similar_ab(self, kernel, feat, dim=0, method="conv"):
        # print("similar_kernel----------", kernel[:], feat[:,0],kernel.shape, feat.shape)
        if method == "similar":
            similar_v = torch.cosine_similarity(kernel, feat, dim=dim)
        elif method == "conv":
            assert kernel.shape[0] == feat.shape[0]
            kernel = kernel.transpose(1, 0)
            kernel = kernel.reshape(-1, feat.shape[0], 1, 1)
            feat = feat.reshape(1, feat.shape[0], feat.shape[1], -1)
            similar_v = F.conv2d(feat, kernel, stride=1)
            if self.cfg.add_focal_loss:
                similar_v += self.KPN_focal_bias
            # similar_v = torch.sigmoid(similar_v)
        # print("similar_v", similar_v, similar_v.shape)
        return similar_v

    def similar_nms(self, score, embedding, nms_th, TOPK_kernel):
        keep = []
        if TOPK_kernel == 0:
            return keep
        order = torch.arange(score.shape[0]).reshape(-1, 1)
        flag = False
        while order.shape[0] > 0:
            # print("order.shape",order.shape)
            idx = order[0, :]
            keep.append(idx)
            if order.shape[0] == 1:
                break
            if len(keep) >= TOPK_kernel:
                break
            cur_kernel = embedding[:, idx].reshape(-1, 1)
            if flag == False:
                cur_embedding = embedding[:, 1:]
                flag = True
            else:
                cur_embedding = cur_embedding[:, 1:]
            #similar = torch.sigmoid(self.similar_ab(cur_kernel, cur_embedding, dim=0).reshape(-1))
            similar = self.similar_ab(cur_kernel, cur_embedding, dim=0).reshape(-1)
            similar = torch.sigmoid(similar)
            # print("similar",similar)

            idxs = torch.where(similar < nms_th)[0]
            # print("idx,similar",idx,similar)
            # print("idxs",idxs.shape)
            # print("______",order,idxs)
            order = order[1 + idxs, :]
            cur_embedding = cur_embedding[:, idxs]
            # print("end------------",idx)
            # print("order after.shape",order.shape)
        if keep == []:
            keep = torch.tensor([])
        else:
            keep = torch.cat(keep, dim=0)
        # print("keep------------", keep, keep.shape)
        return keep

    ############## key_kernel_proposal ###########
    # other version in KPN_backups.py:
    # key_kernel_proposal_train_noNMS() #max pred point in gt_polygon
    # key_kernel_proposal_average()
    #
    ############## key_kernel_proposal ###########
    # @staticmethod
    def get_kernel_cls_mask(self, gt_ys, gt_xs, f_kernel, gt_polygonID_1batch, gt_polygonLable_1batch):
        gt_cls = torch.where(gt_polygonID_1batch > 0, torch.full_like(gt_polygonID_1batch, 1),
                             torch.full_like(gt_polygonID_1batch, 0)).float().detach()
        lst_gt_KPN_cls_1batch = []
        lst_gt_KPN_train_masks_1batch = []
        kernels = f_kernel[:, gt_ys, gt_xs]
        polygonID = gt_polygonID_1batch[:, gt_ys, gt_xs].reshape(-1).detach()
        # polygonLable = gt_polygonLable_1batch[:, gt_ys, gt_xs].detach()
        zeros = torch.full_like(gt_polygonID_1batch, 0).float().detach()
        for i in range(len(polygonID)):
            one_gt = torch.where(gt_polygonID_1batch == polygonID[i], torch.full_like(gt_polygonID_1batch, 1),
                                 torch.full_like(gt_polygonID_1batch, 0)).float().detach()
            ###---lst_gt_KPN_cls
            if polygonID[i] == 0:
                lst_gt_KPN_cls_1batch.append(zeros)
            else:
                lst_gt_KPN_cls_1batch.append(one_gt)
            if self.cfg.cls_branch:
                if self.cfg.dilate_KPN_gt_mask:
                    train_cls_ori = (gt_cls + 0).detach()
                    kernel_size = 40 // self.cfg.scale + 1
                    KPN_train_region = torch.max_pool2d(train_cls_ori, kernel_size=kernel_size, stride=1,
                                                        padding=kernel_size // 2).detach()
            else:
                KPN_train_region = torch.full_like(gt_polygonID_1batch, 1.0).float()

            ###---lst_gt_KPN_train_masks
            if polygonID[i] == 0:
                gt_p_label = 0
            elif polygonID[i] >= len(gt_polygonLable_1batch[0]):
                gt_p_label = 1 #polygon num > cfg.TOPK_kernel
                print("polygonID[i], len(gt_polygonLable_1batch[0])", polygonID[i], len(gt_polygonLable_1batch[0]))
            else:
                gt_p_label = gt_polygonLable_1batch[0][polygonID[i]]

            if gt_p_label == 1 or gt_p_label == 0:
                lst_gt_KPN_train_masks_1batch.append(KPN_train_region)
            elif gt_p_label == self.cfg.DNC_label:  # don't care
                lst_gt_KPN_train_masks_1batch.append((1.0 - one_gt).float() * KPN_train_region)
            else:
                print("in gt gt_p_label=", gt_p_label)
                exit(9)
        gt_cls = torch.cat(lst_gt_KPN_cls_1batch, dim=0)
        gt_mask = torch.cat(lst_gt_KPN_train_masks_1batch, dim=0)
        return kernels, gt_cls, gt_mask

    def key_kernel_proposal(self, pred_cls, nms_score, f_kernel, embedding, gt_centergauss=None,
                            gt_polygonID=None, gt_polygonLable=None, gt_points=None,
                            gt_mask=None, gt_iou_mask=None, istrain=True):
        #############################
        # gt_polygonID: n*1*w*h
        # gt_polygonLable: n*(TOPK+1), [nums, label0, label1...]; +1 store nums
        #                 1: pos; -1: don't care; 0: not gt
        # gt_points: n*2*(TOPK+1), [nums, x0, x1...][nums, y0, y1...]
        #############################
        debug_time = False
        t1 = time.time()
        lst_pred_KPN_cls = []
        lst_gt_KPN_cls = []
        lst_gt_KPN_train_masks = []
        list_gt_kernels = []
        list_pred_xys = []
        if istrain:
            ones_ori = torch.full_like(gt_polygonID[0], 1).float().detach()
            zeros_ = torch.full_like(gt_polygonID[0], 0).float().detach()

        for batch in range(nms_score.shape[0]):
            def get_index_data_inchannel(input, idx):
                out = []
                for c in range(input.shape[0]):
                    out.append(input[c][idx].reshape(1, -1))
                out = torch.cat(out, dim=0)
                return out

            ########### get gt box nums ##########
            box_num = 0
            if istrain:
                box_num1 = gt_polygonLable[batch][0, 0]
                box_num2 = gt_points[batch][0, 0]
                assert box_num1 == box_num2
                box_num = min(box_num1, self.cfg.TOPK_kernel)
            ###########-------end--------get gt box nums ##########

            ########### get gt kernels ##########
            gt_kernels = None
            if istrain and box_num > 0:
                ones_ = ones_ori
                gt_xs = gt_points[batch][0][1:box_num + 1].long()
                gt_ys = gt_points[batch][1][1:box_num + 1].long()
                gt_kernels, gt_cls, gt_mask = self.get_kernel_cls_mask(gt_ys, gt_xs, f_kernel[batch], gt_polygonID[batch],
                                                                   gt_polygonLable[batch])
                list_gt_kernels.append(gt_kernels)
                lst_gt_KPN_cls.append(gt_cls)
                lst_gt_KPN_train_masks.append(gt_mask)

                if self.cfg.KPN_debug:
                    show = torch.where(gt_polygonID[batch, 0] > 0, torch.full_like(gt_polygonID[batch, 0], 255),
                                       torch.full_like(gt_polygonID[batch, 0], 0))
                    show2 = (show + 0).detach().cpu().numpy().astype(np.uint8)
                    # print(gt_xs.shape)
                    show[gt_ys, gt_xs] = 128
                    show = show.detach().cpu().numpy().astype(np.uint8)
                    cv2.namedWindow("show", 0)
                    cv2.imshow("show", show)
                    cv2.namedWindow("show2", 0)
                    cv2.imshow("show2", show2)
                    cv2.waitKey(0)
            ###########-------end-------- get gt kernels##########

            ########### get pred kernels ##########
            t_eval1 = time.time()
            one_pred_center = torch.where(nms_score[batch] >= self.cfg.score_kernel_th, torch.full_like(nms_score[batch], 1),
                                 torch.full_like(nms_score[batch], 0)).detach().cpu().numpy().astype(np.uint8)
            if debug_time:
                print("---KPN GPU2CPU time1", time.time()-t_eval1)
            t_eval1_1 = time.time()
            # nums, labels = cv2.connectedComponents(one_pred_center[0], connectivity=4)
            if istrain:
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(one_pred_center[0], connectivity=4)
                min_areas = stats[:,-1].min()
                topk = max(self.cfg.TOPK_kernel - box_num, 0)
                if nums == 1:
                    topk = 0
                else:
                    topk = topk // (nums-1)
                if min_areas > 0:
                    topk = min(min_areas, topk)
            else:
                nums, labels = cv2.connectedComponents(one_pred_center[0], connectivity=4)
                topk = self.cfg.eval_topk
            t_eval2 = time.time()
            if debug_time:
                print("---KPN get cc time", t_eval2-t_eval1_1)

            labels = to_device(torch.from_numpy(labels)).detach()
            t_nms2_cpu2gpu = time.time()
            t_eval3 = time.time()
            pred_xs = []
            pred_ys = []
            for i in range(1, nums):
                ith_idx = torch.where(labels == i)
                sorted_nms_score, indices = torch.sort(nms_score[batch,0][ith_idx], descending=True)
                indices_topk = indices[0:topk]
                # ith_idx = (ith_idx[0][indices_topk], ith_idx[1][indices_topk])
                if len(indices_topk):
                    pred_xs.append(ith_idx[1][indices_topk])
                    pred_ys.append(ith_idx[0][indices_topk])

            # print(pred_xs, len(pred_xs))
            if len(pred_xs):
                pred_xs = torch.cat(pred_xs)
                pred_ys = torch.cat(pred_ys)
                if istrain:
                    t_nms3 = time.time()
                    pred_kernels, pred_cls, pred_mask = self.get_kernel_cls_mask(pred_ys, pred_xs, f_kernel[batch], gt_polygonID[batch],
                                                                            gt_polygonLable[batch])
                    t_nms4 = time.time()
                    # print("--get gt time, nums:", t_nms4 - t_nms3)
                    list_gt_kernels.append(pred_kernels)
                    lst_gt_KPN_cls.append(pred_cls)
                    lst_gt_KPN_train_masks.append(pred_mask)
                else:
                    pred_kernels = f_kernel[batch][:, pred_ys, pred_xs]
                    list_pred_xys.append(torch.cat([pred_xs.reshape(1, -1), pred_ys.reshape(1, -1)],dim=0))
            else:
                if not istrain:
                    return None, None
                pred_kernels = None
            t_eval4 = time.time()
            if debug_time:
                print("---KPN get kernel time", t_eval4-t_eval3)

            # if batch == 0 and istrain:
            #     self.cfg.iter_show_TOPK_kernel_cnt += 1
            #     if self.cfg.iter_show_TOPK_kernel_cnt >= self.cfg.iter_show_TOPK_kernel:
            #         print("---------post time, nums:", time.time() - t_eval1, nums)
            #         if pred_kernels is not None:
            #             print("pred_kernels.shape", pred_kernels.shape)
            #         else:
            #             print("pred_kernels.shape", 0)
            # ##########-------end--------######### #

            if istrain:
                if gt_kernels is None and pred_kernels is None:
                    selected_kernels = None
                elif pred_kernels is None:
                    selected_kernels = gt_kernels
                elif gt_kernels is None:
                    selected_kernels = pred_kernels
                else:
                    selected_kernels = torch.cat([gt_kernels, pred_kernels], dim=1)
            else:
                selected_kernels = pred_kernels

            # selected_kernels=selected_kernels/(selected_kernels.norm(dim=0,keepdim=True)+0.0000009);
            if selected_kernels is not None:
                t_eval5 = time.time()
                pred_KPN_cls_onebacth = self.similar_ab(selected_kernels, embedding[batch])
                # ##---lst_pred_KPN_cls
                lst_pred_KPN_cls.append(pred_KPN_cls_onebacth)
                pred_KPN_cls = torch.cat(lst_pred_KPN_cls, dim=1)

                t_eval6 = time.time()
                if debug_time:
                    print("---KPN conv time", t_eval6-t_eval5)
            else:
                pred_KPN_cls = None

        if istrain:
            if selected_kernels is not None:
                gt_KPN_cls = torch.cat(lst_gt_KPN_cls, dim=0).unsqueeze(0)
                gt_KPN_train_masks = torch.cat(lst_gt_KPN_train_masks, dim=0).unsqueeze(0)
            else:
                gt_KPN_cls = None
                gt_KPN_train_masks = None

            if self.cfg.train_gt_kernels == False or list_gt_kernels == []:
                list_gt_kernels = None
            # print("gt_KPN_cls, gt_KPN_train_masks", gt_KPN_cls.shape, gt_KPN_train_masks.shape)
            return pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels
        else:
            # pred_xy = torch.cat([pred_ys, pred_xs], dim=0)
            return pred_KPN_cls, list_pred_xys, selected_kernels


    def key_kernel_proposal_old(self, pred_cls, nms_score, f_kernel, embedding, gt_centergauss=None,
                            gt_polygonID=None, gt_polygonLable=None, gt_points=None,
                            gt_mask=None, gt_iou_mask=None, istrain=True):
        #############################
        # gt_polygonID: n*1*w*h
        # gt_polygonLable: n*(TOPK+1), [nums, label0, label1...]; +1 store nums
        #                 1: pos; -1: don't care; 0: not gt
        # gt_points: n*2*(TOPK+1), [nums, x0, x1...][nums, y0, y1...]
        #############################
        lst_pred_KPN_cls = []
        lst_gt_KPN_cls = []
        lst_gt_KPN_train_masks = []
        list_gt_kernels = []
        if istrain:
            ones_ori = torch.full_like(gt_polygonID[0], 1).float().detach()
            zeros_ = torch.full_like(gt_polygonID[0], 0).float().detach()

        for batch in range(nms_score.shape[0]):
            def get_index_data_inchannel(input, idx):
                out = []
                for c in range(input.shape[0]):
                    out.append(input[c][idx].reshape(1, -1))
                out = torch.cat(out, dim=0)
                return out

            ########### get gt box nums ##########
            box_num = 0
            if istrain:
                box_num1 = gt_polygonLable[batch][0, 0]
                box_num2 = gt_points[batch][0, 0]
                assert box_num1 == box_num2
                box_num = min(box_num1, self.cfg.TOPK_kernel)
            ###########-------end--------##########

            ########### get gt kernels ##########
            if istrain:
                ones_ = ones_ori
                gt_xs = gt_points[batch][0][1:box_num + 1].long()
                gt_ys = gt_points[batch][1][1:box_num + 1].long()
                gt_kernels, gt_cls, gt_mask = self.get_kernel_cls_mask(gt_ys, gt_xs, f_kernel[batch], gt_polygonID[batch], gt_polygonLable[batch])
                list_gt_kernels.append(gt_kernels)
                lst_gt_KPN_cls.append(gt_cls)
                lst_gt_KPN_train_masks.append(gt_mask)

                if self.cfg.KPN_debug:
                    show = torch.where(gt_polygonID[batch,0] > 0, torch.full_like(gt_polygonID[batch,0], 255),
                                             torch.full_like(gt_polygonID[batch,0], 0))
                    show2 = (show + 0).detach().cpu().numpy().astype(np.uint8)
                    #print(gt_xs.shape)
                    show[gt_ys, gt_xs] = 128
                    show = show.detach().cpu().numpy().astype(np.uint8)
                    cv2.namedWindow("show", 0)
                    cv2.imshow("show", show)
                    cv2.namedWindow("show2", 0)
                    cv2.imshow("show2", show2)
                    cv2.waitKey(0)
            ###########-------end--------##########

            ########### get pred kernels ##########
            score_th_idx = torch.where(nms_score[batch] >= self.cfg.score_kernel_th)
            # print("score_th_idx", score_th_idx)
            if len(score_th_idx[0]) == 0:
                score_th_idx = (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
            score_th_idx_wh = (score_th_idx[1], score_th_idx[2])

            sorted_nms_score, indices = torch.sort(nms_score[batch][score_th_idx], descending=True)
            embedding_one = get_index_data_inchannel(f_kernel[batch], score_th_idx_wh)
            embedding_one = embedding_one[:, indices]

            if not istrain:
                print("sorted_nms_score.shape", sorted_nms_score.shape)
            t_nms1 = time.time()
            keep_num = self.cfg.TOPK_kernel - box_num
            keep = self.similar_nms(sorted_nms_score, embedding_one, nms_th=self.cfg.nms_th, TOPK_kernel=keep_num)
            t_nms2 = time.time()
            if batch == 0 and istrain:
                self.cfg.iter_show_TOPK_kernel_cnt += 1
                if self.cfg.iter_show_TOPK_kernel_cnt >= self.cfg.iter_show_TOPK_kernel:
                    print("---------mns time, keep TOPK_kernel:", t_nms2 - t_nms1, keep, len(keep))
            pred_kernrls = embedding_one[:, keep]

            if istrain:
                gt_polygonID_one = get_index_data_inchannel(gt_polygonID[batch], score_th_idx_wh)
                gt_polygonID_one = gt_polygonID_one[:, indices]
                for k in keep:
                    boxid = gt_polygonID_one[0, k]
                    if boxid == 0:
                        one_gt = zeros_
                    else:
                        one_gt = torch.where(gt_polygonID[batch] == boxid, torch.full_like(gt_polygonID[batch], 1),
                                             torch.full_like(gt_polygonID[batch], 0)).float().detach()
                    ###---lst_gt_KPN_cls
                    lst_gt_KPN_cls.append(one_gt)

                    if self.cfg.cls_branch:
                        if self.cfg.dilate_KPN_gt_mask:
                            train_cls_ori = (gt_cls + 0).detach()
                            kernel_size = 40 // self.cfg.scale + 1
                            KPN_train_region = torch.max_pool2d(train_cls_ori, kernel_size=kernel_size, stride=1,
                                                                padding=kernel_size // 2).detach()
                    else:
                        KPN_train_region = 1.0

                    ###---lst_pred_KPN_cls
                    if boxid == 0:
                        #print("in pred gt_p_label=", gt_p_label)
                        lst_gt_KPN_train_masks.append(ones_.float() * KPN_train_region)
                        #exit(9)
                    else:
                        gt_p_label = gt_polygonLable[batch, 0][boxid]
                        if gt_p_label == 1:
                            lst_gt_KPN_train_masks.append(ones_.float() * KPN_train_region)
                        elif gt_p_label == self.cfg.DNC_label:  # don't care
                            lst_gt_KPN_train_masks.append((ones_ - one_gt).float() * KPN_train_region)
            ###########-------end--------##########


            if istrain:
                selected_kernels = torch.cat([gt_kernels, pred_kernrls], dim=1)
            else:
                selected_kernels = pred_kernrls
            # selected_kernels=selected_kernels/(selected_kernels.norm(dim=0,keepdim=True)+0.0000009);
            pred_KPN_cls_onebacth = self.similar_ab(selected_kernels, embedding[batch])
            ###---lst_pred_KPN_cls
            lst_pred_KPN_cls.append(pred_KPN_cls_onebacth)


        if istrain:
            pred_KPN_cls = torch.cat(lst_pred_KPN_cls, dim=1)
            gt_KPN_cls = torch.cat(lst_gt_KPN_cls, dim=0).unsqueeze(0)
            gt_KPN_train_masks = torch.cat(lst_gt_KPN_train_masks, dim=0).unsqueeze(0)
            return pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels
        else:
            pred_KPN_cls = torch.cat(lst_pred_KPN_cls, dim=1)
            return pred_KPN_cls

    def forward(self, x, gt_polygonID=None, gt_polygonLable=None, gt_points=None, gt_centergauss=None,
                                gt_mask=None, gt_iou_mask=None, istrain=True):
        t0 = time.time()
        if self.cfg.scale == 1:
            feat_used, up2, up3, up4, up5 = self.fpn(x)
        elif self.cfg.scale == 2:
            feat_used, up3, up4, up5 = self.fpn(x)

        # print(feat_used.shape, up3.shape, up4.shape, up5.shape)
        b_time = time.time()-t0
        
        if not istrain:
            x = to_device(torch.Tensor(2))
            x = x.cpu()

        t1 = time.time()
        pred_cls, pred_centergauss, f_kernel, pred_embedding = self.feature(feat_used) #f_cls, f_offset, f_embedding
        if self.cfg.cls_branch:
            pred_nms_score = (F.sigmoid(pred_cls).detach() * F.sigmoid(pred_centergauss).detach()).detach()
            pred_cls_score = F.sigmoid(pred_cls).detach()
        else:
            pred_nms_score = F.sigmoid(pred_centergauss).detach()
            pred_cls_score = None

        # print("---time KPN head:,", time.time()-t1)
        t2 = time.time()
        if istrain:
            pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels = \
                self.key_kernel_proposal(pred_cls_score, pred_nms_score, f_kernel, pred_embedding, gt_centergauss,
                                         gt_polygonID=gt_polygonID, gt_polygonLable=gt_polygonLable, gt_points=gt_points,
                                         gt_mask=gt_mask, gt_iou_mask=gt_iou_mask, istrain=istrain)
            #print(pred_KPN_cls.shape, gt_KPN_cls.shape, gt_KPN_train_masks.shape)
            IM_time = time.time()-t1
            return pred_cls, pred_centergauss, pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels#, b_time, IM_time
        else:
            pred_KPN_cls, list_pred_xys, selected_kernels = \
                self.key_kernel_proposal(pred_cls_score, pred_nms_score, f_kernel, pred_embedding, istrain=istrain)
            IM_time = time.time()-t1
            # print("---time KPN:,", time.time()-t2)
            return pred_cls, pred_centergauss, pred_KPN_cls, list_pred_xys, b_time, IM_time



if __name__ == "__main__":
    import numpy as np
    score = np.array([[[[0.1, 0.6, 0.5], [0.2, 0.7, 0.1], [0.9, 0.7, 0.1]]],
                      [[[0.2, 0.2, 0.8], [0.2, 0.2, 0.1], [0.9, 0.1, 0.1]]]])
    embedding = np.array([[[[1, 2, 3], [19, 18, 71], [29, 18, 7]],
                           [[1, 2, 3], [39, 82, 47], [1, 2, 3]],
                           [[49, 81, 17], [59, 28, 7], [1, 2, 3]]],
                          [[[1, 2, 3], [1, 2, 3], [69, 28, 37]],
                           [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                           [[79, 18, 37], [1, 2, 3], [1, 2, 3]]]])

    embedding = np.transpose(embedding, (0, 3, 1, 2))

    score = np.random.rand(3, 1, 1000, 1000)
    embedding = np.random.randn(3, 9, 1000, 1000)
    # print("embedding",embedding)
    gt_polygonID = score + 1
    print(score.shape)
    print(embedding.shape)
    score = torch.from_numpy(score).float()
    embedding = torch.from_numpy(embedding).float()
    gt_polygonID = torch.from_numpy(gt_polygonID).float()



    from util.config import config as cfg, update_config, print_config
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)

    points = key_kernel_proposal(score, embedding, gt_polygonID, istrain=False)


    if self.cfg.debug_synth:
        feat_used = self.debug_conv31(x)
        pred_cls = None
        pred_centergauss = self.debug_conv11(feat_used)
        f_kernel = self.debug_conv11(feat_used)
        pred_embedding = self.debug_conv11(feat_used)
        t1 = time.time()
        if istrain:
            pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels = \
                self.key_kernel_proposal(torch.sigmoid(pred_centergauss), torch.sigmoid(pred_centergauss), f_kernel,
                                         pred_embedding, gt_centergauss,
                                         gt_polygonID=gt_polygonID, gt_polygonLable=gt_polygonLable,
                                         gt_points=gt_points,
                                         gt_mask=gt_mask, gt_iou_mask=gt_iou_mask, istrain=istrain)
            # print(pred_KPN_cls.shape, gt_KPN_cls.shape, gt_KPN_train_masks.shape)
            IM_time = time.time() - t1
            #return pred_cls, pred_centergauss, pred_KPN_cls, gt_KPN_cls, gt_KPN_train_masks, list_gt_kernels  # , b_time, IM_time
