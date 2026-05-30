import torch
import torch.nn as nn
import random
import math

################################################################################
# Moiré attention에서 사용할 focus function
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
# 노이즈, 드롭아웃, FFN 모듈
################################################################################

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        # 학습 중에만 작은 Gaussian noise를 추가하여 overfitting을 완화
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
# Moiré Attention + SEE logit injection 레이어
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

        # 각 attention head가 집중할 거리 구간을 학습할 수 있도록 shift와 width를 parameter로 둠
        # shape: (1, num_heads, 1, 1)로 만들어 batch/node 차원에 broadcast되도록 합니다.
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )
        
        # self-loop 위치에 더해줄 head별 bias
        self.self_loop_W = nn.Parameter(
            torch.tensor(
                [1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                dtype=torch.float,
            ).view(1, num_heads, 1, 1),
            requires_grad=False,
        )

        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)

        # 전처리에서 계산된 SEE edge_attr를 모델 내부에서 사용할 수 있는 형태로 변환
        # SEE score 계산 자체는 parameter-free이지만, 이 FFN은 task-specific edge score 보정을 담당합니다.
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

        # edge_attr는 전처리에서 계산된 edge-level spectral sensitivity
        # edge_ffn을 거쳐 모델의 task에 맞게 edge score를 보정
        edge_attr = self.edge_ffn(edge_attr)  # (B, E, 1)

        # 현재 edge_attr_dim=1
        # 이후 모든 attention head에 동일한 SEE score를 전달
        edge_attr = edge_attr.squeeze(-1).unsqueeze(1)  # (B, 1, E)
        edge_attr = edge_attr.expand(-1, self.num_heads, -1)  # (B, H, E)
        
        # head별 learnable weight를 곱해 각 head가 SEE를 반영하는 강도를 다르게 학습
        weighted_edge_attr = edge_attr * self.edge_weight_per_head.view(1, self.num_heads, 1)

        # 기존의 모아레 어텐션 가중치 계산 유지
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2


        # adj의 거리값을 focus function에 통과시켜 거리 기반 Moiré attention bias를 만듬
        # log를 취한 뒤 attention logit에 더하면 softmax 이전 단계에서 multiplicative bias처럼 동작
        # head별로 다른 거리 focus bias를 만드는 부분
        moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(min=1e-6)
        adjusted_scores = scores + torch.log(moire_adj)


        # SEE score를 attention logit에 주입
        # 각 edge가 어느 batch sample에 속하는지 지정 
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1) # (B, 1, 1)
        batch_indices = batch_indices.expand(-1, self.num_heads, edge_index.size(-1)) # (B, H, E)


        # 각 edge score를 어느 attention head에 더할지 지정
        head_indices = torch.arange(self.num_heads, device=x.device).view(1, -1, 1) # (1, H, 1)
        head_indices = head_indices.expand(batch_size, -1, edge_index.size(-1)) #  (B, H, E)

        # edge_index의 source(u)/target(v) node index를 head 차원으로 broadcast
        # attention score 행렬에서 row/col 위치
        edge_index_u = edge_index[:, 0, :].unsqueeze(1).expand(-1, self.num_heads, -1) # (B, H, E)
        edge_index_v = edge_index[:, 1, :].unsqueeze(1).expand(-1, self.num_heads, -1) # (B, H, E)

        # adjusted_scores에 SpectralEdge Score 추가로 적용
        # 이 연산 이후 softmax가 적용되므로, SEE가 큰 edge는 attention probability가 커지는 방향으로 작용
        adjusted_scores[batch_indices, head_indices, edge_index_u, edge_index_v] += weighted_edge_attr # (B, H, N, N)

        # self-loop 추가
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)

        # padding node가 attention에 참여하지 않도록 logit을 큰 음수로 masking
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)

        # softmax attention 및 output projection
        # see score가 큰 edge는 softmax 이후 attention probability가 증가하는 방향으로 작용
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

        # head별로 서로 다른 거리 영역을 보도록 초기 shift를 무작위로 설정
        # width는 shift에 따라 지수적으로 설정하여 focus 범위를 조절
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
            edge_attr_dim,
        )

        # 비선형적으로 변환 FFN
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
        # Moiré attention + SEE logit injection
        h = self.attention(x, adj, edge_index, edge_attr, mask)

        # 마스크 적용 (노드가 유효하지 않다면 0으로)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # FFN 적용
        h = self.ffn(h)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))

        # Residual Connection
        # attention-FFN 결과와 input projection을 0.5:0.5 비율로 섞습니다.
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h

