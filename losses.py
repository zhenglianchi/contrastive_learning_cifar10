import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module): # inspired by : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.06, device="cuda:0"): # after hyperparam optimization, temperature of 0.06 gave the best ICBHI score
        super().__init__()
        #temperature对loss进行缩放
        self.temperature = temperature
        self.device = device

    def forward(self, projection1, projection2, labels=None):
        
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        #特征拼接
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        #batch_size
        batch_size = features.shape[0]
        #lables和mask不可以同时存在，labels存在则为有监督对比学习
        #mask是将labels中属于同一类的变成1例如torch.tensor([1,1,3]) -> mask = tensor([[1., 1., 0.],[1., 1., 0.],[0., 0., 1.]])
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        #两个数据增强
        contrast_count = features.shape[1]
        #拆分tensor，每个张量为一个列表再次拼接得到(2 * batch_size, feature_size)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #计算了所有 features 两两之间的余弦相似度(内积)得到（2 * batch_size，2 * batch_size]）
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        #记录了每一行的最大值。 也就是与每个样本有最大多少的相似度。(最大不超过1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #广播相减,(由于后面要进行指数操作,因此全负数为了数值稳定性。这个计算在loge方运算中不改变最后的计算值)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability
        #为了完全对齐相应logits的形状 得到(2 * batch_size, 2 * batch_size)
        mask = mask.repeat(contrast_count, contrast_count)
        #logits_mask = （2 * batch_size, 2 * batch_size）
        #其中全矩阵值为1， 而对角线值为0。logits_mask 用于保证每个feature不与自己本身进行对比
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # mask = mask * logits_mask之后， mask中所有非自己的同类别样本上都是1。
        mask = mask * logits_mask

        #所有的正样本指数运算后的值
        exp_logits = torch.exp(logits) * logits_mask
        #以下为公式
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss

class SupConCELoss(nn.Module):
    def __init__(self, alpha=0.5, device="cuda:0", temperature=0.06): # after hyperparam optimization, loss tradeoff of 0.5 gave the best score
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature, device=device)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, projection1, projection2, prediction1, prediction2, target):

        predictions = torch.cat([prediction1, prediction2], dim=0)
        target_ = torch.nn.functional.one_hot(target, num_classes=10).float()
        labels = torch.cat([target_, target_], dim=0)
        return self.alpha * self.supcon(projection1, projection2, target) + (1 - self.alpha) * self.ce(predictions, labels)