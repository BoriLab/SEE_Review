
import torch
import torch.nn as nn
import random
import math

################################################################################
# 1. 모아레 어텐션에서 사용할 집중(focus) 함수들
################################################################################

def gaussian_attention(distances, shift, width):
    # 최소값을 clamp로 제한하여 width=0이 되지 않도록 방지
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
# 3. 모아레 어텐션 레이어 (for-문 제거 버전)
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

        self.edge_ffn = FFN(edge_attr_dim, edge_attr_dim, edge_attr_dim)

        self.scale2 = math.sqrt(self.head_dim)

        # 추가된 부분: ESS를 위한 헤드별 가중치 파라미터 정의
        self.edge_weight_per_head = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x, adj, edge_index, edge_attr, mask):
        batch_size, num_nodes, _ = x.size()

        # Q, K, V 생성
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 엣지 속성 FFN 적용
        # edge_attr = self.edge_ffn(edge_attr) 
        edge_attr = self.edge_ffn(edge_attr)  # (batch_size, num_edges, 1)

        # 🔥 SpectralEdge Score 처리 (차원 수정됨)
        edge_attr = edge_attr.squeeze(-1).unsqueeze(1)  # (batch_size, 1, num_edges)
        edge_attr = edge_attr.expand(-1, self.num_heads, -1)  # (batch_size, num_heads, num_edges)
        
        weighted_edge_attr = edge_attr * self.edge_weight_per_head.view(1, self.num_heads, 1)

        # 기존의 모아레 어텐션 가중치 계산 유지
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2
        moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(min=1e-6)
        adjusted_scores = scores + torch.log(moire_adj)

        # 엣지 인덱스 관련 고급 인덱싱
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand(-1, self.num_heads, edge_index.size(-1))

        head_indices = torch.arange(self.num_heads, device=x.device).view(1, -1, 1)
        head_indices = head_indices.expand(batch_size, -1, edge_index.size(-1))

        edge_index_u = edge_index[:, 0, :].unsqueeze(1).expand(-1, self.num_heads, -1)
        edge_index_v = edge_index[:, 1, :].unsqueeze(1).expand(-1, self.num_heads, -1)

        # 🔥 adjusted_scores에 SpectralEdge Score 추가로 적용 🔥
        adjusted_scores[batch_indices, head_indices, edge_index_u, edge_index_v] += weighted_edge_attr

        # self-loop 추가
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)

        # 마스크 처리
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)

        # Attention weights 및 최종 출력
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
        focus,         # 예: get_moire_focus("gaussian") 등
        edge_attr_dim, # 엣지 속성 차원
    ):
        super(MoireLayer, self).__init__()

        # 초기 shift와 width를 무작위로 설정
        shifts = [
            shift_min + random.uniform(0, 1) * (shift_max - shift_min)#random.uniform(0, 1)
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
            edge_attr_dim,  # 추가된 부분
        )

        # FFN
        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)

        # 입력 차원을 residual에 맞추기 위한 projection
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask):
        """
        x: (batch_size, num_nodes, input_dim)
        adj: (batch_size, num_nodes, num_nodes)
        edge_index: (batch_size, 2, num_edges)
        edge_attr: (batch_size, num_edges, edge_attr_dim)
        mask: (batch_size, num_nodes)
        """
        # 모아레 어텐션
        h = self.attention(x, adj, edge_index, edge_attr, mask)

        # 마스크 적용 (노드가 유효하지 않다면 0으로)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # FFN 적용
        h = self.ffn(h)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # Residual Connection
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h

