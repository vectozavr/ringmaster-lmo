import os
import ssl
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_tinyshakespeare_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        try:
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, data_path)
        except Exception:
            ssl_context = ssl._create_unverified_context()
            with urllib.request.urlopen(TINY_SHAKESPEARE_URL, context=ssl_context) as response:
                with open(data_path, "wb") as output:
                    output.write(response.read())
    return data_path


class CharacterTokenizer:
    def __init__(self, text):
        self._chars = sorted(set(text))
        self.stoi = {char: idx for idx, char in enumerate(self._chars)}
        self.itos = {idx: char for char, idx in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self._chars)

    def encode(self, text):
        return [self.stoi[char] for char in text]


@dataclass
class NanochatConfig:
    # very small model for testing purposes; not intended to produce good results
    '''
    sequence_len: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    '''
    sequence_len: int = 2048
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanochatGPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.sequence_len, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class NanochatLanguageModelFunction:
    def __init__(
        self,
        data_dir,
        config=None,
        seed=0,
        batch_size=8,
        eval_batch_size=8,
        eval_batches=8,
        val_fraction=0.1,
        is_cuda=False,
    ):
        self._generator = np.random.default_rng(seed)
        self._torch_generator = torch.Generator()
        self._torch_generator.manual_seed(seed)
        if is_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if config is None:
            config = NanochatConfig()
        self.config = config
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batches = eval_batches

        data_path = ensure_tinyshakespeare_dataset(data_dir)
        with open(data_path, "r", encoding="utf-8") as source:
            text = source.read()
        self.tokenizer = CharacterTokenizer(text)
        token_ids = np.array(self.tokenizer.encode(text), dtype=np.int64)
        split_index = int(len(token_ids) * (1.0 - val_fraction))
        self.train_tokens = torch.tensor(token_ids[:split_index], dtype=torch.long, device=self.device)
        self.val_tokens = torch.tensor(token_ids[split_index:], dtype=torch.long, device=self.device)

        self._model = NanochatGPT(config, self.tokenizer.vocab_size).to(self.device)
        self._parameter_infos = self._build_parameter_infos()
        self._eval_starts = self._make_eval_starts()

    def value(self, point):
        self._set_point(point)
        self._model.eval()
        losses = []
        with torch.no_grad():
            for starts in self._eval_starts:
                x, y = self._batch_from_starts(self.val_tokens, starts)
                losses.append(self._model(x, y).detach().cpu().item())
        return float(np.mean(losses))

    def gradient(self, point):
        self._set_point(point)
        self._model.train()
        starts = self._make_train_starts(self.batch_size)
        x, y = self._batch_from_starts(self.train_tokens, starts)
        self._model.zero_grad(set_to_none=True)
        loss = self._model(x, y)
        loss.backward()
        return self._parameters_to_numpy(grad=True)

    def stochastic_gradient(self, point):
        return self.gradient(point)

    def dim(self):
        return sum(parameter.numel() for parameter in self._parameters_iter())

    def get_current_point(self):
        return self._parameters_to_numpy(grad=False)

    def parameter_metadata(self):
        return {"parameter_infos": self._parameter_infos}

    def _parameters_iter(self):
        for parameter in self._model.parameters():
            yield parameter

    def _parameters_to_numpy(self, grad=False):
        flat_parameters = []
        for parameter in self._parameters_iter():
            tensor = parameter.grad if grad else parameter.detach()
            flat_parameters.append(tensor.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=True))
        return np.concatenate(flat_parameters)

    def _set_point(self, point):
        point = np.asarray(point, dtype=np.float32)
        shift = 0
        with torch.no_grad():
            for parameter in self._parameters_iter():
                num_elements = parameter.numel()
                values = torch.from_numpy(point[shift:shift + num_elements]).view_as(parameter).to(self.device)
                parameter.copy_(values)
                shift += num_elements

    def _build_parameter_infos(self):
        parameter_infos = []
        for name, parameter in self._model.named_parameters():
            use_muon = (
                parameter.ndim >= 2
                and "token_embedding" not in name
                and "position_embedding" not in name
                and "lm_head" not in name
            )
            parameter_infos.append({
                "name": name,
                "shape": list(parameter.shape),
                "use_muon": use_muon,
            })
        return parameter_infos

    def _make_eval_starts(self):
        max_start = len(self.val_tokens) - self.config.sequence_len - 1
        if max_start <= 0:
            raise ValueError("Validation split is too small for the configured sequence length.")
        return [
            self._generator.integers(0, max_start, size=self.eval_batch_size)
            for _ in range(self.eval_batches)
        ]

    def _make_train_starts(self, batch_size):
        max_start = len(self.train_tokens) - self.config.sequence_len - 1
        if max_start <= 0:
            raise ValueError("Training split is too small for the configured sequence length.")
        return self._generator.integers(0, max_start, size=batch_size)

    def _batch_from_starts(self, tokens, starts):
        inputs = []
        targets = []
        seq_len = self.config.sequence_len
        for start in starts:
            start = int(start)
            chunk = tokens[start:start + seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
        x = torch.stack(inputs)
        y = torch.stack(targets)
        return x, y
