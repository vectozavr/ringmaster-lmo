import math
import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
DEFAULT_EVAL_TOKENS = 40 * 524288

CACHE_DIR = os.environ.get("AUTORESEARCH_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "autoresearch"))
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
DEPTH = 8


def download_single_shard(index):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{BASE_URL}/{filename}"
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    temp_path = filepath + ".tmp"
    with open(temp_path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    os.replace(temp_path, filepath)
    return True


def download_data(num_shards):
    num_train = min(num_shards, MAX_SHARD)
    shard_ids = list(range(num_train))
    if VAL_SHARD not in shard_ids:
        shard_ids.append(VAL_SHARD)
    for index in shard_ids:
        download_single_shard(index)


def list_parquet_files():
    if not os.path.exists(DATA_DIR):
        return []
    files = sorted(name for name in os.listdir(DATA_DIR) if name.endswith(".parquet") and not name.endswith(".tmp"))
    return [os.path.join(DATA_DIR, name) for name in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    parquet_paths = [path for path in list_parquet_files() if not path.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + idx for idx, name in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as output:
        pickle.dump(encoding, output)

    token_bytes = []
    special_set = set(SPECIAL_TOKENS)
    for token_id in range(encoding.n_vocab):
        token_str = encoding.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), token_bytes_path)


def ensure_autoresearch_assets(num_shards):
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        return
    download_data(num_shards)
    train_tokenizer()


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as source:
            enc = pickle.load(source)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as source:
        return torch.load(source, map_location=device)


def _document_batches(split):
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise RuntimeError("No parquet shards found. Preparation failed.")
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [path for path in parquet_paths if path != val_path]
        if not parquet_paths:
            raise RuntimeError("No training shards found after preparation.")
    else:
        parquet_paths = [val_path]

    epoch = 1
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx)
                batch = row_group.column("text").to_pylist()
                yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, batch_size, sequence_len, split, buffer_size=1000, device="cuda"):
    row_capacity = sequence_len + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        texts, epoch = next(batches)
        token_lists = tokenizer.encode(texts, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda idx: len(doc_buffer[idx]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        inputs = row_buffer[:, :-1].to(device=device, non_blocking=False)
        targets = row_buffer[:, 1:].to(device=device, non_blocking=False)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, eval_tokens, device):
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    steps = max(1, eval_tokens // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


@dataclass
class NanochatConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def build_model_config(depth, vocab_size, sequence_len=MAX_SEQ_LEN, aspect_ratio=ASPECT_RATIO,
                       head_dim=HEAD_DIM, window_pattern=WINDOW_PATTERN):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return NanochatConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


def _build_window_mask(seq_len, window_size, device):
    left_window = window_size[0]
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    if left_window < 0 or left_window >= seq_len:
        return distance >= 0
    return (distance >= 0) & (distance < left_window)


def _load_flash_attention_interface():
    try:
        from kernels import get_kernel
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    try:
        return get_kernel(repo).flash_attn_interface
    except Exception:
        return None


FLASH_ATTN_INTERFACE = _load_flash_attention_interface()


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        batch_size, seq_len, _ = x.size()
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)

        if FLASH_ATTN_INTERFACE is not None and x.is_cuda:
            y = FLASH_ATTN_INTERFACE.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_head != self.n_head:
                repeat_factor = self.n_head // self.n_kv_head
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            attn_mask = _build_window_mask(seq_len, window_size, q.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            y = y.transpose(1, 2)

        y = y.contiguous().view(batch_size, seq_len, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class NanochatGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for value_embed in self.value_embeds.values():
            torch.nn.init.uniform_(value_embed.weight, -scale, scale)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=self.transformer.wte.weight.device)
        self.cos = cos
        self.sin = sin
        if self.transformer.wte.weight.is_cuda:
            self.transformer.wte.to(dtype=torch.bfloat16)
            for value_embed in self.value_embeds.values():
                value_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[layer_idx % len(pattern)]] for layer_idx in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.size()
        cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for layer_idx, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            ve = self.value_embeds[str(layer_idx)](idx) if str(layer_idx) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[layer_idx])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits


class NanochatLanguageModelFunction:
    def __init__(
        self,
        seed=42,
        is_cuda=True,
        num_shards=None,
        device_batch_size=2,
        eval_batch_size=2,
        eval_tokens=None,
        depth=DEPTH,
        aspect_ratio=ASPECT_RATIO,
        head_dim=HEAD_DIM,
        window_pattern=WINDOW_PATTERN,
        compile_model=None,
    ):
        if is_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.seed = seed
        self.device_batch_size = device_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_tokens = eval_tokens if eval_tokens is not None else int(
            os.environ.get("AUTORESEARCH_EVAL_TOKENS", str(2 * MAX_SEQ_LEN * eval_batch_size))
        )
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.set_float32_matmul_precision("high")

        prepare_shards = num_shards if num_shards is not None else int(os.environ.get("AUTORESEARCH_NUM_SHARDS", "2"))
        ensure_autoresearch_assets(prepare_shards)

        self.tokenizer = Tokenizer.from_directory()
        self.config = build_model_config(
            depth=depth,
            vocab_size=self.tokenizer.get_vocab_size(),
            sequence_len=MAX_SEQ_LEN,
            aspect_ratio=aspect_ratio,
            head_dim=head_dim,
            window_pattern=window_pattern,
        )

        self._model = NanochatGPT(self.config).to(self.device)
        self._model.init_weights()

        should_compile = compile_model
        if should_compile is None:
            should_compile = self.device.type == "cuda" and os.environ.get("RINGMASTER_TORCH_COMPILE", "0") == "1"
        if should_compile:
            self._model = torch.compile(self._model, dynamic=False)

        self._parameter_infos = self._build_parameter_infos()
        self._train_loader = make_dataloader(
            self.tokenizer,
            self.device_batch_size,
            self.config.sequence_len,
            "train",
            device=self.device,
        )

    def value(self, point):
        self._set_point(point)
        self._model.eval()
        return float(evaluate_bpb(
            self._model,
            self.tokenizer,
            batch_size=self.eval_batch_size,
            eval_tokens=self.eval_tokens,
            device=self.device,
        ))

    def gradient(self, point):
        self._set_point(point)
        self._model.train()
        x, y, _ = next(self._train_loader)
        self._model.zero_grad(set_to_none=True)
        with self.autocast_ctx:
            loss = self._model(x, y)
        loss.backward()
        return self._parameters_to_flat_tensor(grad=True)

    def stochastic_gradient(self, point):
        return self.gradient(point)

    def dim(self):
        return int(sum(parameter.numel() for parameter in self._parameters_iter()))

    def get_current_point(self):
        return self._parameters_to_flat_tensor(grad=False)

    def parameter_metadata(self):
        return {
            "parameter_infos": self._parameter_infos,
            "model_config": asdict(self.config),
        }

    def _parameters_iter(self):
        for parameter in self._model.parameters():
            yield parameter

    def _parameters_to_flat_tensor(self, grad=False):
        chunks = []
        for parameter in self._parameters_iter():
            tensor = parameter.grad if grad else parameter.detach()
            if tensor is None:
                tensor = torch.zeros_like(parameter)
            chunks.append(tensor.reshape(-1).float())
        return torch.cat(chunks)

    def _set_point(self, point):
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point = point.to(device=self.device, dtype=torch.float32)

        shift = 0
        with torch.no_grad():
            for parameter in self._parameters_iter():
                num_elements = parameter.numel()
                values = point[shift:shift + num_elements].view_as(parameter).to(dtype=parameter.dtype)
                parameter.copy_(values)
                shift += num_elements

    def _build_parameter_infos(self):
        infos = []
        for name, parameter in self._model.named_parameters():
            use_muon = name.startswith("transformer.h") and parameter.ndim >= 2
            infos.append({
                "name": name,
                "shape": list(parameter.shape),
                "use_muon": use_muon,
            })
        return infos
