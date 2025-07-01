import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import re
from transformers import AutoModel # Added for gpt2 backbone

# --- Constants ---
VALID_NEURON_SELECT_TYPES = ['first-last', 'random', 'random-pairing']
VALID_BACKBONE_TYPES = [
    f'resnet{depth}-{i}' for depth in [18, 34, 50, 101, 152] for i in range(1, 5)
] + ['shallow-wide', 'parity_backbone', 'gpt2', 'mnist', 'minigrid', 'classic_control', 'qamnist_operator', 'qamnist_index'] # Added missing backbone types
VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier', 'multi-learnable-fourier-1d', # Added missing positional embedding
    'custom-rotational', 'custom-rotational-1d', 'sinusoidal' # Added missing positional embedding
]

# --- Utils ---
def compute_decay(T, params, clamp_lims=(0, 15)):
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'
    
    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    B, H, W = x.shape
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    coords = torch.stack((x_coords, y_coords), dim=-1)
    coords = coords.unsqueeze(0)
    coords = coords.repeat(B, 1, 1, 1)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    preds = F.softmax(logits, dim=-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(preds * log_preds, dim=-1)
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)
    return normalized_entropy

# --- Modules ---
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        return x.squeeze(self.dim)

class SynapseUNET(nn.Module):
    def __init__(self,
                 out_dims,
                 depth,
                 minimum_width=16,
                 dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth
        widths = np.linspace(out_dims, minimum_width, depth)
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])),
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )
        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1
        for i in range(num_blocks):
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        out_first = self.first_projection(x)
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))
        outs_up = outs_down[-1]
        num_blocks = len(self.up_projections)
        for i in range(num_blocks):
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)
        return outs_up

class SuperLinear(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        self.register_parameter('T', nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        out = self.dropout(x)
        out = self.layernorm(out)
        out = torch.einsum('BODM,MHD->BODH', out, self.w1) + self.b1
        out = out.squeeze(-1) / self.T
        return out

class ParityBackbone(nn.Module):
    def __init__(self, n_embeddings, d_embedding):
        super(ParityBackbone, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        x = (x == 1).long()
        return self.embedding(x.long()).transpose(1, 2)

class QAMNISTOperatorEmbeddings(nn.Module):
    def __init__(self, num_operator_types, d_projection):
        super(QAMNISTOperatorEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_operator_types, d_projection)

    def forward(self, x):
        return self.embedding(-x - 1)

class QAMNISTIndexEmbeddings(torch.nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        embedding = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return self.embedding[x]
    
class ThoughtSteps:
    def __init__(self, iterations_per_digit, iterations_per_question_part, total_iterations_for_answering, total_iterations_for_digits, total_iterations_for_question):
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.total_iterations_for_answering = total_iterations_for_answering
        self.total_iterations_for_digits = total_iterations_for_digits
        self.total_iterations_for_question = total_iterations_for_question
        self.total_iterations = self.total_iterations_for_digits + self.total_iterations_for_question + self.total_iterations_for_answering

    def determine_step_type(self, stepi: int):
        is_digit_step = stepi < self.total_iterations_for_digits
        is_question_step = self.total_iterations_for_digits <= stepi < self.total_iterations_for_digits + self.total_iterations_for_question
        is_answer_step = stepi >= self.total_iterations_for_digits + self.total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_answer_step_type(self, stepi: int):
        step_within_questions = stepi - self.total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

class MNISTBackbone(nn.Module):
    def __init__(self, d_input):
        super(MNISTBackbone, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.layers(x)

class MiniGridBackbone(nn.Module):
    def __init__(self, d_input, grid_size=7, num_objects=11, num_colors=6, num_states=3, embedding_dim=8):
        super().__init__()
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        self.position_embedding = nn.Embedding(grid_size * grid_size, embedding_dim)
        self.project_to_d_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        x = x.long()
        B, H, W, C = x.size()
        object_idx = x[:,:,:, 0]
        color_idx =  x[:,:,:, 1]
        state_idx =  x[:,:,:, 2]
        obj_embed = self.object_embedding(object_idx)
        color_embed = self.color_embedding(color_idx)
        state_embed = self.state_embedding(state_idx)
        pos_idx = torch.arange(H * W, device=x.device).view(1, H, W).expand(B, -1, -1)
        pos_embed = self.position_embedding(pos_idx)
        out = self.project_to_d_projection(torch.cat([obj_embed, color_embed, state_embed, pos_embed], dim=-1))
        return out

class ClassicControlBackbone(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        return self.input_projector(x)

class ShallowWide(nn.Module):
    def __init__(self):
        super(ShallowWide, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(4096, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
            nn.BatchNorm2d(2048),
            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1, groups=32),
            nn.GLU(dim=1),
            nn.BatchNorm2d(2048)
        )
    def forward(self, x):
        return self.layers(x)

class PretrainedResNetWrapper(nn.Module):
    def __init__(self, resnet_type, fine_tune=True):
        super(PretrainedResNetWrapper, self).__init__()
        self.resnet_type = resnet_type
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', resnet_type, pretrained=True)
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()

    def forward(self, x):
        out = self.backbone(x)
        nc = 256 if ('18' in self.resnet_type or '34' in self.resnet_type) else 512 if '50' in self.resnet_type else 1024 if '101' in self.resnet_type else 2048
        num_features = out.shape[-1]
        wh_squared = num_features / nc
        if wh_squared < 0 or not float(wh_squared).is_integer():
             print(f"Warning: Cannot reliably reshape PretrainedResNetWrapper output. nc={nc}, num_features={num_features}")
             return out
        wh = int(np.sqrt(wh_squared))
        return out.reshape(x.size(0), nc, wh, wh)

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma=1/2.5,
                 ):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = d_model
        self.gamma = gamma
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GLU(),
            nn.Linear(self.H_dim // 2, self.D // self.G),
            nn.LayerNorm(self.D // self.G)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_coord = add_coord_dim(x[:,0])
        projected = self.Wr(x_coord)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = (1.0 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1)
        Y = self.mlp(F)
        PEx = Y.permute(0, 3, 1, 2)
        return PEx

class MultiLearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma_range=[1.0, 0.1],
                 N=10,
                 ):
        super().__init__()
        self.embedders = nn.ModuleList()
        for gamma in np.linspace(gamma_range[0], gamma_range[1], N):
            self.embedders.append(LearnableFourierPositionalEncoding(d_model, G, M, F_dim, H_dim, gamma))
        self.register_parameter('combination', torch.nn.Parameter(torch.ones(N), requires_grad=True))
        self.N = N

    def forward(self, x):
        pos_embs = torch.stack([emb(x) for emb in self.embedders], dim=0)
        weights = F.softmax(self.combination, dim=-1).view(self.N, 1, 1, 1, 1)
        combined_emb = (pos_embs * weights).sum(0)
        return combined_emb

class MultiLearnableFourierPositionalEncoding1D(nn.Module):
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma_range=[1.0, 0.1],
                 N=10,
                 ):
        super().__init__()
        self.embedders = nn.ModuleList()
        for gamma in np.linspace(gamma_range[0], gamma_range[1], N):
            self.embedders.append(LearnableFourierPositionalEncoding(d_model, G, M, F_dim, H_dim, gamma))
        self.register_parameter('combination', torch.nn.Parameter(torch.ones(N), requires_grad=True))
        self.N = N
        self.d_model = d_model

    def forward(self, x):
        x_reshaped = x.permute(0, 2, 1).unsqueeze(2)
        pos_embs = torch.stack([emb(x_reshaped) for emb in self.embedders], dim=0)
        weights = F.softmax(self.combination, dim=-1).view(self.N, 1, 1, 1, 1)
        combined_emb = (pos_embs * weights).sum(0)
        output = combined_emb.squeeze(2).permute(0, 2, 1)
        return output

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        elif x.dim() == 2:
             x = x.unsqueeze(1)
             x = x + self.pe[:x.size(0)]
             x = x.squeeze(1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CustomRotationalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding, self).__init__()
        self.register_parameter('start_vector', nn.Parameter(torch.Tensor([0, 1]), requires_grad=True))
        self.projection = nn.Sequential(nn.Linear(4, d_model))

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        theta_rad = torch.deg2rad(torch.linspace(0, 180, W, device=device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=1)
        rotated_vectors = torch.einsum('wij,j->wi', rotation_matrices, self.start_vector)
        key = torch.cat((
            torch.repeat_interleave(rotated_vectors.unsqueeze(1), W, dim=1),
            torch.repeat_interleave(rotated_vectors.unsqueeze(0), W, dim=0)
        ), dim=-1)
        pe_grid = self.projection(key)
        pe = pe_grid.permute(2, 0, 1).unsqueeze(0)
        if H != W:
            pass
        return pe

class CustomRotationalEmbedding1D(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding1D, self).__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x):
        start_vector = torch.tensor([0., 1.], device=x.device, dtype=torch.float)
        theta_rad = torch.deg2rad(torch.linspace(0, 180, x.size(2), device=x.device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_theta = cos_theta.unsqueeze(1)
        sin_theta = sin_theta.unsqueeze(1)
        rotation_matrices = torch.stack([
        torch.cat([cos_theta, -sin_theta], dim=1),
        torch.cat([sin_theta, cos_theta], dim=1)
        ], dim=1)
        rotated_vectors = torch.einsum('bij,j->bi', rotation_matrices, start_vector)
        pe = self.projection(rotated_vectors)
        pe = torch.repeat_interleave(pe.unsqueeze(0), x.size(0), 0)
        return pe.transpose(1, 2)

# --- ResNet ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_scales,
        stride,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        do_initial_max_pool=True,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        ) if in_channels in [1, 3] else nn.LazyConv2d(
            self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if do_initial_max_pool else Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_scales = feature_scales
        if 2 in feature_scales:
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=stride, dilate=replace_stride_with_dilation[0]
            )
            if 3 in feature_scales:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=stride, dilate=replace_stride_with_dilation[1]
                )
                if 4 in feature_scales:
                    self.layer4 = self._make_layer(
                        block, 512, layers[3], stride=stride, dilate=replace_stride_with_dilation[2]
                    )
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if 2 in self.feature_scales:
            x = self.layer2(x)
            if 3 in self.feature_scales:
                x = self.layer3(x)
                if 4 in self.feature_scales:
                    x = self.layer4(x)
        return x

def _resnet(in_channels, feature_scales, stride, arch, block, layers, pretrained, progress, device, do_initial_max_pool, **kwargs):
    model = ResNet(in_channels, feature_scales, stride, block, layers, do_initial_max_pool=do_initial_max_pool, **kwargs)
    if pretrained:
        assert in_channels==3
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + '/state_dicts/' + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet18(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    return _resnet(in_channels,
        feature_scales, stride, "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, do_initial_max_pool, **kwargs
    )

def resnet34(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    return _resnet(in_channels,
        feature_scales, stride, "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )

def resnet50(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    return _resnet(in_channels,
        feature_scales, stride, "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )

def prepare_resnet_backbone(backbone_type):
    resnet_family = resnet18
    if '34' in backbone_type: resnet_family = resnet34
    if '50' in backbone_type: resnet_family = resnet50
    block_num_str = backbone_type.split('-')[-1]
    hyper_blocks_to_keep = list(range(1, int(block_num_str) + 1)) if block_num_str.isdigit() else [1, 2, 3, 4]
    backbone = resnet_family(
        3,
        hyper_blocks_to_keep,
        stride=2,
        pretrained=False,
        progress=True,
        device="cpu",
        do_initial_max_pool=True,
    )
    return backbone

# --- CTM ---
class ContinuousThoughtMachine(nn.Module):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 context_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super(ContinuousThoughtMachine, self).__init__()
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.context_dims = context_dims
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        self.memory_length = memory_length
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm
        self.verify_args()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(self.context_dims)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((self.context_dims, d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((self.context_dims, d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == 'action':
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        
        if self.neuron_select_type in ('first-last', 'random'):
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action':
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out':
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, :, neuron_indices_left]
                selected_right = activated_state[:, :, neuron_indices_right]
            outer = selected_left.unsqueeze(-1) * selected_right.unsqueeze(-2)
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, :, i, j]
            
        elif self.neuron_select_type == 'random-pairing':
            left = activated_state[:, :, neuron_indices_left]
            right = activated_state[:, :, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")
        
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        self.kv_features = self.backbone(x)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        B, C = current_prediction.size(0), current_prediction.size(1)
        reshaped_pred = current_prediction.reshape([B, C] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred, 'none')
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def get_d_backbone(self):
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        elif self.backbone_type == 'gpt2':
            return 768 # Default hidden size for gpt2
        elif self.backbone_type == 'mnist':
            return self.d_input # MNIST backbone projects to d_input
        elif self.backbone_type == 'minigrid':
            return self.d_input # MiniGrid backbone projects to d_input
        elif self.backbone_type == 'classic_control':
            return self.d_input # ClassicControl backbone projects to d_input
        elif self.backbone_type == 'qamnist_operator':
            return self.d_input # QAMNIST operator embedding projects to d_input
        elif self.backbone_type == 'qamnist_index':
            return self.d_input # QAMNIST index embedding projects to d_input
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'gpt2':
            self.embedding_model = AutoModel.from_pretrained('gpt2')
            self.backbone = nn.Sequential(
                self.embedding_model.get_input_embeddings()
            )
        elif self.backbone_type == 'mnist':
            self.backbone = MNISTBackbone(self.d_input)
        elif self.backbone_type == 'minigrid':
            self.backbone = MiniGridBackbone(self.d_input)
        elif self.backbone_type == 'classic_control':
            self.backbone = ClassicControlBackbone(self.d_input)
        elif self.backbone_type == 'qamnist_operator':
            self.backbone = QAMNISTOperatorEmbeddings(num_operator_types=2, d_projection=self.d_input) # Assuming 2 operator types for QAMNIST
        elif self.backbone_type == 'qamnist_index':
            self.backbone = QAMNISTIndexEmbeddings(max_seq_length=100, embedding_dim=self.d_input) # Assuming max_seq_length 100 for QAMNIST
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'multi-learnable-fourier-1d':
            return MultiLearnableFourierPositionalEncoding1D(d_backbone)
        elif self.positional_embedding_type == 'sinusoidal':
            return SinusoidalPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
            assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
            synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
            self.register_buffer(f'{synch_type}_neuron_indices_left', left)
            self.register_buffer(f'{synch_type}_neuron_indices_right', right)
            self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(self.context_dims, synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)
        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))
        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    def verify_args(self):
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"
        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size

    def forward(self, x, track=False, padding_mask=None):
        B = x.size(0)
        device = x.device
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        kv = self.compute_features(x)
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1, -1)
        predictions = torch.empty(B, self.context_dims, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, self.context_dims, 2, self.iterations, device=device, dtype=torch.float32)
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        for stepi in range(self.iterations):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')
            q = self.q_proj(synchronisation_action)
            attn_out, attn_weights = self.attention(q, kv, kv,
                                                    average_attn_weights=False,
                                                    need_weights=True,
                                                    key_padding_mask=padding_mask,
                                                    is_causal=True)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)
            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out