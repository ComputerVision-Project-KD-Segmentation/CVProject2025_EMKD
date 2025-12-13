import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module) :
    def __init__(self, gamma=0, alpha=None, size_average=True) :
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)) :
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list) :
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target) :
        if input.dim() > 2 :
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target[:, 1 :].contiguous()
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None :
            if self.alpha.type() != input.data.type() :
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).to(torch.int64))
            logpt = logpt * Variable(at)

        loss = -(1 - pt) ** self.gamma * logpt
        if self.size_average :
            return loss.mean()
        else :
            return loss.sum()


def dice_loss(prediction, target) :
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    prediction = torch.softmax(prediction, dim=1)[:, 1:].contiguous()
    target = target[:, 1:].contiguous()

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, ce_weight=0.5) :
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        ce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """

    focal_loss = FocalLoss(gamma=2, alpha=torch.FloatTensor([1., 1.]))
    ce = focal_loss(prediction, target)

    dice = dice_loss(prediction, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    return loss


def dice_score(prediction, target) :
    prediction = torch.sigmoid(prediction)
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()
    return (2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)


def prediction_map_distillation(y, teacher_scores, T=4) :
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    # Student와 Teacher의 Height(H) 크기 비교
    s_H = s.shape[2]
    t_H = t.shape[2]
    # print("t shape :", s_H)
    # print("s shape :", t_H)
    
    # Case 1: Student가 더 크면 -> Student를 줄여서 Teacher에 맞춤
    if s_H > t_H:
        # s = F.interpolate(s, t.shape[-2:], mode='bilinear', align_corners=False)
        avg_pool = nn.AdaptiveAvgPool2d((t_H, t_H))
        s = avg_pool(s)
    # Case 2: Teacher가 더 크면 -> Teacher를 줄여서 Student에 맞춤
    elif t_H > s_H:
        # t = F.interpolate(t, s.shape[-2:], mode='bilinear', align_corners=False)
        avg_pool = nn.AdaptiveAvgPool2d((s_H, s_H))
        t = avg_pool(t)
    
    # print("s interpolate shape :", s.shape)
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    eps = 1e-6
    mask0 = gt[:, 0].unsqueeze(1)
    mask1 = gt[:, 1].unsqueeze(1)

    # Add eps to denominators to prevent NaN
    region0 = torch.sum(x * mask0, dim=(2, 3)) / (torch.sum(mask0, dim=(2, 3)) + eps)
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + eps)
    
    return F.cosine_similarity(region0, region1, dim=1)


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    # 1. Resize GT to match Student spatial dimensions
    # Use mode='nearest' to preserve integer class values if gt is discrete, 
    # or 'bilinear' if gt is soft probabilities.
    # gt_s = F.interpolate(gt, size=s.shape[2:], mode='nearest')
    gt_s = F.interpolate(gt, size=s.shape[2:], mode='bilinear', align_corners=False)
    gt_s = torch.clamp(gt_s, 0, 1)
    rc_s = region_contrast(s, gt_s)

    # 2. Resize GT to match Teacher spatial dimensions
    # gt_t = F.interpolate(gt, size=t.shape[2:], mode='nearest')
    gt_t = F.interpolate(gt, size=t.shape[2:], mode='bilinear', align_corners=False)
    gt_t = torch.clamp(gt_t, 0, 1)
    rc_t = region_contrast(t, gt_t)

    return (rc_s - rc_t).pow(2).mean()

# ==================== EDGE Knowledge Distillation Losses ====================

def extract_edge(mask, kernel_size=3):
    """
    Extract edges from binary/one-hot mask using convolution
    Args:
        mask: (B, C, H, W)
        kernel_size: convolution kernel size
    """
    B, C, H, W = mask.shape
    device = mask.device

    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    padding = kernel_size // 2

    edge_masks = []
    for c in range(C):
        channel_mask = mask[:, c:c + 1]  # (B, 1, H, W)

        conv_result = F.conv2d(channel_mask.float(), kernel, padding=padding)

        max_val = kernel_size * kernel_size
        edge = (conv_result > 0) & (conv_result < max_val)
        edge_masks.append(edge.float())

    return torch.cat(edge_masks, dim=1)


class BoundaryKDV1(nn.Module):
    """
    Boundary Knowledge Distillation (ECKD component)
    Based on EDGE: Edge Constraint Knowledge Distillation
    """
    def __init__(self, kernel_size=3, tau=4, num_classes=2,
                 one_hot_target=True, include_background=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.tau = tau
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background

    def forward(self, preds_S, preds_T, gt_labels):
        """
        Args:
            preds_S: Student logits (B, C, Hs, Ws)
            preds_T: Teacher logits (B, C, Ht, Wt)
            gt_labels: one-hot GT (B, C, Hg, Wg)
        """
        # 1) spatial 정렬 (모두 GT 크기로 맞추기)
        target_size = gt_labels.shape[2:]
        if preds_S.shape[2:] != target_size:
            preds_S = F.interpolate(preds_S, size=target_size,
                                    mode='bilinear', align_corners=False)
        if preds_T.shape[2:] != target_size:
            preds_T = F.interpolate(preds_T, size=target_size,
                                    mode='bilinear', align_corners=False)

        # 2) edge mask 추출
        edge_mask = extract_edge(gt_labels, self.kernel_size)

        if not self.include_background:
            # background 채널 제외 (class 0)
            edge_mask = edge_mask[:, 1:]
            preds_S = preds_S[:, 1:]
            preds_T = preds_T[:, 1:]

        # 3) edge 영역만 사용
        preds_S_edge = preds_S * edge_mask
        preds_T_edge = preds_T * edge_mask

        # 4) KL-divergence on edge regions
        p = F.log_softmax(preds_S_edge / self.tau, dim=1)
        q = F.softmax(preds_T_edge / self.tau, dim=1)

        loss = F.kl_div(p, q, reduction='batchmean') * (self.tau ** 2)
        return loss


class LogHausdorffDTLoss(nn.Module):
    """
    Log Hausdorff Distance Transform Loss (SRKD component)
    Based on EDGE: Segmentation Refinement KD
    """
    def __init__(self, alpha=2.0, include_background=False,
                 to_onehot_y=True, sigmoid=False, softmax=True):
        super().__init__()
        self.alpha = alpha
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax

    def distance_field(self, img):
        """
        Simplified distance transform for 2D images
        Args:
            img: Binary mask (B, C, H, W)
        Returns:
            Distance field (B, C, H, W)
        """
        from scipy.ndimage import distance_transform_edt
        import numpy as np

        device = img.device
        field = torch.zeros_like(img)

        for b in range(len(img)):
            for c in range(img.shape[1]):
                mask_np = img[b, c].cpu().numpy() > 0.5

                if mask_np.any() and not mask_np.all():
                    fg_dist = distance_transform_edt(mask_np)
                    bg_dist = distance_transform_edt(~mask_np)
                    field[b, c] = torch.from_numpy(fg_dist + bg_dist).float()

        return field.to(device)

    def forward(self, preds_S, preds_T, target):
        """
        Args:
            preds_S: Student logits/prob (B, C, Hs, Ws)
            preds_T: Teacher logits/prob (B, C, Ht, Wt)
            target: GT or pseudo-GT (B, C, Ht, Wt) / (B, 1, Ht, Wt)
        """
        # 일단 spatial size를 target 기준으로 맞추는 게 가장 직관적이지만,
        # DSDLoss 쪽에서 이미 teacher / label 기준으로 맞춰줄 것이므로
        # 여기서는 기준이 target이라고 가정.
        target_size = target.shape[2:]
        if preds_S.shape[2:] != target_size:
            preds_S = F.interpolate(preds_S, size=target_size,
                                    mode='bilinear', align_corners=False)
        if preds_T.shape[2:] != target_size:
            preds_T = F.interpolate(preds_T, size=target_size,
                                    mode='bilinear', align_corners=False)

        if self.sigmoid:
            preds_S = torch.sigmoid(preds_S)

        if self.softmax and preds_S.shape[1] > 1:
            preds_S = torch.softmax(preds_S, dim=1)

        # Teacher를 pseudo label로 사용
        if self.to_onehot_y:
            preds_T = preds_T.detach()
            target = preds_T.argmax(dim=1, keepdim=True)  # (B,1,H,W)
            target_onehot = torch.zeros_like(preds_T)
            target_onehot.scatter_(1, target, 1)
            target = target_onehot

        if not self.include_background and preds_S.shape[1] > 1:
            preds_S = preds_S[:, 1:]
            target = target[:, 1:]

        losses = []
        for c in range(preds_S.shape[1]):
            pred_c = preds_S[:, c:c + 1]
            target_c = target[:, c:c + 1]

            with torch.no_grad():
                pred_dt = self.distance_field(pred_c.detach())
                target_dt = self.distance_field(target_c.detach())

            pred_error = (pred_c - target_c) ** 2
            distance = pred_dt ** self.alpha + target_dt ** self.alpha

            loss_c = (pred_error * distance).mean()
            losses.append(loss_c)

        loss = torch.stack(losses).mean()
        log_loss = torch.log(loss + 1)
        return log_loss


class DSDLoss(nn.Module):
    """
    Deep Supervision Distillation Loss (EDGE complete framework)
    Combines:
      - ECKD (BoundaryKD)
      - SRKD (LogHausdorffDT)
      - SAKD (Multi-scale via projector)
    """
    def __init__(self, in_chans, num_classes, num_stages, cur_stage,
                 kernel_size=3, interpolate=False,
                 bd_include_background=False, hd_include_background=False,
                 one_hot_target=True, sigmoid=False, softmax=True, tau=4,
                 loss_weight=1.0, overall_loss_weight=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.interpolate = interpolate  # (이제는 사용 안 해도 되지만, 인터페이스는 유지)
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.overall_loss_weight = overall_loss_weight

        # Projector: student feature -> logits space
        if cur_stage != num_stages:
            # Multi-stage: upsampling blocks
            up_sample_blk_num = num_stages - cur_stage
            layers = []

            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # 마지막 블록: num_classes로 projection
                    layers.append(
                        nn.ConvTranspose2d(
                            in_chans, num_classes,
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1, bias=True
                        )
                    )
                else:
                    out_chans = in_chans // 2
                    layers.extend([
                        nn.ConvTranspose2d(
                            in_chans, out_chans,
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1, bias=True
                        ),
                        nn.InstanceNorm2d(out_chans, affine=True),
                        nn.PReLU()
                    ])
                    in_chans = out_chans

            self.projector = nn.Sequential(*layers)
        else:
            # 마지막 스테이지: 1x1 conv 또는 identity
            if num_classes == in_chans:
                self.projector = nn.Identity()
            else:
                self.projector = nn.Conv2d(in_chans, num_classes, 1, 1, 0, bias=True)

        # projector 초기화
        for m in self.projector.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ECKD
        self.bkd = BoundaryKDV1(
            kernel_size=kernel_size,
            tau=tau,
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background
        )

        # SRKD
        self.hd = LogHausdorffDTLoss(
            include_background=hd_include_background,
            to_onehot_y=one_hot_target,
            sigmoid=sigmoid,
            softmax=softmax
        )

    def forward(self, feat_student, logits_teacher, label):
        """
        Args:
            feat_student: 학생 feature (B, C, Hs, Ws)
            logits_teacher: teacher logits (B, num_classes, Ht, Wt)
            label: GT one-hot (B, num_classes, Hl, Wl)
        """
        # 1) student feature -> logits space
        logits_student = self.projector(feat_student)

        # 2) 모든 logits를 label 크기에 맞추기 (EDGE 모든 컴포넌트 일관성 유지)
        target_size = label.shape[2:]  # (H, W)

        if logits_student.shape[2:] != target_size:
            logits_student = F.interpolate(
                logits_student, size=target_size,
                mode='bilinear', align_corners=False
            )

        if logits_teacher.shape[2:] != target_size:
            logits_teacher = F.interpolate(
                logits_teacher, size=target_size,
                mode='bilinear', align_corners=False
            )

        # 3) ECKD: Edge-constrained KD
        bkd_loss = self.bkd(
            preds_S=logits_student,
            preds_T=logits_teacher,
            gt_labels=label
        )

        # 4) SRKD: Hausdorff Distance 기반 refinement
        hd_loss = self.hd(
            preds_S=logits_student,
            preds_T=logits_teacher,
            target=label
        )

        # 5) 가중합
        bkd_loss = bkd_loss * self.overall_loss_weight * self.loss_weight
        hd_loss = hd_loss * self.overall_loss_weight * (1 - self.loss_weight)

        return {
            'bkd_loss': bkd_loss,
            'hd_loss': hd_loss
        }