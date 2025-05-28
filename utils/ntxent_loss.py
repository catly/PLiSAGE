import torch
import torch.nn as nn
class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)
        self.mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        # 创建对角线掩码
        N = 2 * self.batch_size
        mask = torch.ones((N, N), device=self.device).bool()
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        # 直接使用嵌入
        z = torch.cat((z_i, z_j), dim=0)

        # 计算余弦相似度矩阵
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # 从相似度矩阵中提取正样本对的相似度
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)

        # 应用掩码以提取负样本
        negative_samples = sim[self.mask].view(N, -1)

        # 创建标签张量
        labels = torch.zeros(N, device=self.device).long()

        # 合并正样本和负样本并计算损失
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss / N


