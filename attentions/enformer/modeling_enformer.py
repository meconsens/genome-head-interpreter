#replace: https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py

import math
import torch
from torch import nn #, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot

from enformer_pytorch.config_enformer import EnformerConfig
from torch.cuda.amp import GradScaler, autocast

from transformers import PreTrainedModel

# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

def collect_attentions(module, attentions=None):
    if attentions is None:
        attentions = []
        
    for child in module.children():
        if isinstance(child, Residual) and child.attn is not None:
            attentions.append(child.attn)
        else:
            collect_attentions(child, attentions)
    return attentions


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

# relative positional encoding functions

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


            
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length
        self.original_seq_len = None  # Add this line

    def forward(self, x):
        #print("IN TARGET LENGTH CROP FORWARD, X IS: ", x.shape)
        seq_len, target_len = x.shape[-2], self.target_length
        
        self.original_seq_len = seq_len

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def ConvBlock(dim, dim_out = None, kernel_size = 1):
    #print("ENTERING CONV BLOCK")
    return Sequential(
        BatchNorm1d(dim),
        GELU(),
        Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        output_attentions=True,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.attentions = None
        self.output_attentions =True
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = Linear(dim, dim_key * heads, bias = False)
        self.to_k = Linear(dim, dim_key * heads, bias = False)
        self.to_v = Linear(dim, dim_value * heads, bias = False)

        self.to_out = Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.q_content_bias = Add()
        self.q_position_bias = Add()
        self.logits = Add()
        self.q_scale = ElementMul()

        # dropouts

        self.pos_dropout = Dropout(pos_dropout)
        self.attn_dropout = Dropout(dropout)

        self.q = None
        self.k = None
        self.q = None
        self.positions = None
        self.einsum1 = None
        self.einsum2 = None 
        self.einsum3 = None
        self.rel_k = None
        self.lrp_score_matrix = None
        self.lrp_scores = None

    def forward(self, x):
       #print("IN ATTENTION FORWARD, X IS: ", x.shape)
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = self.q_scale(q,self.scale)

        einsum_obj1 = einsum('b h i d, b h j d -> b h i j')
        q_content_bias = self.q_content_bias([q, self.rel_content_bias])
        content_logits = einsum_obj1(q_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        self.positions = positions

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        einsum_obj2 = einsum('b h i d, h j d -> b h i j')
       
        q_position_bias = self.q_position_bias([q, self.rel_pos_bias])
        
        rel_logits = einsum_obj2(q_position_bias, rel_k)
       
        rel_logits = relative_shift(rel_logits)
       

        logits = self.logits([content_logits,rel_logits])
        attn = logits.softmax(dim = -1)
        self.attentions = attn
        attn = self.attn_dropout(attn)
        
        einsum_obj3 = einsum('b h i j, b h j d -> b h i d')
       
        out = einsum_obj3(attn, v)
       
        out = rearrange(out, 'b h n d -> b n (h d)')
       

        self.einsum1= einsum_obj1
        self.einsum2= einsum_obj2
        self.einsum3= einsum_obj3
        self.rel_k = rel_k
        self.q = q 
        self.k = k 
        self.v = v


        if self.output_attentions:
            return self.to_out(out), self.attentions
        else:
            return self.to_out(out)


class SequentialAttention(nn.Module):
    def __init__(self, *args):
        super(SequentialAttention, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            
    def forward(self, x):
        for module in self._modules.values():
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x
    




class DropoutAttention(nn.Module):
    def __init__(self, p: float = 0.5):
        super(DropoutAttention, self).__init__()
        self.p = p

    def forward(self, *inputs):
        if self.training:
            if isinstance(inputs[0], tuple):
                return (F.dropout(inputs[0][0], self.p, self.training),) + inputs[0][1:]
            else:
                return (F.dropout(inputs[0], self.p, self.training),)
        return inputs

# main class

class Enformer(PreTrainedModel):
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = EnformerConfig()

        # make sure that the base_model_prefix attribute is defined
        assert hasattr(cls, "base_model_prefix"), f"Attribute `base_model_prefix` is not defined for model class {cls}"

        model = cls(config, *model_args, **kwargs)
        state_dict = torch.load(pretrained_model_name_or_path)
        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config, output_attentions=True):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        self.num_layers = config.depth
        self.num_heads = config.heads

        # create stem

        self.stem = Sequential(
            Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        #create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = Sequential(*conv_layers)

        transformer = []
        for _ in range(config.depth):
            transformer.append(SequentialAttention(
                Residual(SequentialAttention(
                    LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.dim // config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = config.dim // config.heads,
                        output_attentions = output_attentions # add output_attentions here
                    ),
                DropoutAttention(config.dropout_rate)
            )),
            Residual(SequentialAttention(
                LayerNorm(config.dim),
                Linear(config.dim, config.dim * 2),
                DropoutAttention(config.dropout_rate),
                ReLU(),
                Linear(config.dim * 2, config.dim),
                DropoutAttention(config.dropout_rate)
            ))
        ))

        self.transformer = SequentialAttention(*transformer)


        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            Dropout(config.dropout_rate / 8),
            GELU()
        )

        # create trunk sequential module

        self._trunk = Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: Sequential(
            Linear(self.dim * 2, features),
            Softplus()
        ), kwargs))

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    #@profile
    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None
    ):
        with autocast():
            if isinstance(x, list):
                x = str_to_one_hot(x)

            elif x.dtype == torch.long:
                x = seq_indices_to_one_hot(x)

            no_batch = x.ndim == 2

            if no_batch:
                x = rearrange(x, '... -> () ...')

            if exists(target_length):
                self.set_target_length(target_length)

            trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
            x = trunk_fn(x)

            if no_batch:
                x = rearrange(x, '() ... -> ...')

            if return_only_embeddings:
                return x

            out = map_values(lambda fn: fn(x), self._heads)

            if exists(head):
                assert head in self._heads, f'head {head} not found'
                out = out[head]

            if exists(target):
                assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

                if return_corr_coef:
                    return pearson_corr_coef(out, target)

                return poisson_loss(out, target)

            if return_embeddings:
                return out, x

            #if output_attentions flag is passed, set the flag in attention layer.
            if self.output_attentions:
                attentions = collect_attentions(self.transformer)
        
            if self.output_attentions:
                return out, attentions
            else:
                return out
