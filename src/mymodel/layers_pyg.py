import torch
import torch.nn as nn
import random
import math


################################################################################
# 1. 모아레 어텐션에서 사용할 집중(focus) 함수들
################################################################################

def gaussian_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-((distances - shift) ** 2) / width)

def laplacian_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-torch.abs(distances - shift) / width)

def cauchy_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + ((distances - shift) / width) ** 2)

def sigmoid_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + torch.exp((-distances + shift) / width))

def triangle_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.clamp(1 - torch.abs(distances - shift) / width, min=0)

def get_moire_focus(attention_type):
    if attention_type == "gaussian":
        return gaussian_attention
    elif attention_type == "laplacian":
        return laplacian_attention
    elif attention_type == "cauchy":
        return cauchy_attention
    elif attention_type == "sigmoid":
        return sigmoid_attention
    elif attention_type == "triangle":
        return triangle_attention
    else:
        raise ValueError("Invalid attention type")

################################################################################
# 2. 노이즈, 드롭아웃, FFN 모듈
################################################################################

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            dropout_mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * dropout_mask
        return x

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            GaussianNoise(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.ffn(x)


################################################################################
# 3. 모아레 어텐션 레이어 (SEE 제거, PyG edge_attr 사용)
################################################################################
class MoireAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        initial_shifts,
        initial_widths,
        focus,
        edge_attr_dim,
    ):
        super(MoireAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert (
            self.head_dim * num_heads == output_dim
        ), "output_dim must be divisible by num_heads"

        self.focus = focus
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )

        self.self_loop_W = nn.Parameter(
            torch.tensor(
                [1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                dtype=torch.float,
            ).view(1, num_heads, 1, 1),
            requires_grad=False,
        )

        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)

        # PyG에서 제공되는 edge_attr를 변환하기 위한 FFN
        # (예: bond type 등 범주형 -> one-hot으로 들어올 수도 있고,
        #  혹은 float일 수 있으므로, 상황에 맞게 차원 지정)
        self.edge_ffn = FFN(edge_attr_dim, edge_attr_dim, 1)  # 예: 최종 1차원으로

        self.scale2 = math.sqrt(self.head_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask):
        """
        x: (batch_size, num_nodes, input_dim)
        adj: (batch_size, num_nodes, num_nodes)
        edge_index: (batch_size, 2, num_edges)
        edge_attr: (batch_size, num_edges, edge_attr_dim)
        mask: (batch_size, num_nodes)
        """
        batch_size, num_nodes, _ = x.size()

        # 1) Q, K, V 생성
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2) PyG 엣지속성 -> FFN
        #    (batch_size, num_edges, edge_attr_dim) -> (batch_size, num_edges, 1)
        edge_attr = self.edge_ffn(edge_attr)  # 예: 최종 1차원
        edge_attr = edge_attr.squeeze(-1)     # shape: (batch_size, num_edges)

        # 3) Focus (모아레) adj
        #    distances = adj, shifts=..., widths=...
        moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(min=1e-6)

        # 4) scores: QK^T / sqrt(...)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2

        # 5) batch_indices & head_indices for scatter
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand(-1, self.num_heads, edge_index.size(-1))

        head_indices = torch.arange(self.num_heads, device=x.device).view(1, -1, 1)
        head_indices = head_indices.expand(batch_size, -1, edge_index.size(-1))

        edge_index_u = edge_index[:, 0, :].unsqueeze(1).expand(-1, self.num_heads, -1)
        edge_index_v = edge_index[:, 1, :].unsqueeze(1).expand(-1, self.num_heads, -1)

        # 6) adjusted_scores = scores + log(moire_adj) + edge_attr
        adjusted_scores = scores + torch.log(moire_adj)

        #    edge_attr를 scores에 반영
        #    edge_attr.shape = (batch_size, num_edges)
        #    => 인덱싱 & broadcast
        #    edge_attr_ij = edge_attr[batch, edge_idx]
        edge_attr_val = edge_attr[batch_indices, edge_index_v]  # 예시: v노드 또는 u노드. 필요에 따라 조정 가능
        # (위에서 edge_attr를 (u,v) 중 어느 것에 매핑할지 결정해야 합니다.
        #  보통 edge는 u->v나 v->u 동일하므로, 어느쪽 쓰든 상관없을 때가 많음.)

        adjusted_scores[batch_indices, head_indices, edge_index_u, edge_index_v] += edge_attr_val

        # 7) self-loop
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)

        # 8) 마스크 처리
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)

        # 9) Attention weights & output
        attention_weights = torch.softmax(adjusted_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        return out


################################################################################
# 4. 모아레 레이어
################################################################################

class MoireLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        shift_min,
        shift_max,
        dropout,
        focus,
        edge_attr_dim,  # ← PyG의 원본 edge_attr 차원
    ):
        super(MoireLayer, self).__init__()

        # 초기 shift, width
        shifts = [
            shift_min + random.uniform(0, 1) * (shift_max - shift_min)
            for _ in range(num_heads)
        ]
        widths = [1.3 ** shift for shift in shifts]

        # 모아레 어텐션 모듈
        self.attention = MoireAttention(
            input_dim,
            output_dim,
            num_heads,
            shifts,
            widths,
            focus,
            edge_attr_dim,  # ← SEE 대신, 원본 edge_attr
        )

        # FFN
        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)

        # 입력 차원을 residual에 맞추기 위한 projection
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask):
        """
        x:           (batch_size, num_nodes, input_dim)
        adj:         (batch_size, num_nodes, num_nodes)
        edge_index:  (batch_size, 2, num_edges)
        edge_attr:   (batch_size, num_edges, edge_attr_dim)  # 원본 PyG 엣지 특성
        mask:        (batch_size, num_nodes)
        """
        # 1) 모아레 어텐션
        h = self.attention(x, adj, edge_index, edge_attr, mask)

        # 2) 마스크
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # 3) FFN
        h = self.ffn(h)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # 4) Residual
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h
