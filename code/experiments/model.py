import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class SimpleNeuralNetFunction:
    """
    Public API:
      - value(point) -> numpy scalar loss (full batch)
      - gradient(point) -> flat numpy gradient (full batch by default; see flag)
      - stochastic_gradient(point, batch_size=None, idx=None, replace=False) -> flat numpy grad on a minibatch
      - stochastic_gradient_at_points(points, batch_size=None, idx=None, replace=False) -> list of minibatch grads
      - dim() -> number of parameters
      - get_current_point() -> current params as flat numpy vector
      - _loss(point) -> torch scalar loss (full batch)
      - _logits(point) -> torch logits (full batch)
      - _check_accuracy(point) -> float accuracy in [0,1]
    """

    def __init__(self, features, labels, number_of_classes=10, is_cuda=False,
                 reg_parameter=0.0, activation="relu", hidden_dim=128,
                 default_batch_size=4, seed=None,
                 use_minibatch_for_gradient=False):
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # device & data
        self.device = torch.device("cuda" if (is_cuda and torch.cuda.is_available()) else "cpu")
        self.X = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.y_np = np.asarray(labels)
        self.y = torch.tensor(labels, dtype=torch.long, device=self.device)
        self.N = self.X.shape[0]

        # shapes
        self.input_dim = self.X.shape[1]
        self.num_classes = number_of_classes
        self.hidden_dim = hidden_dim

        # model (inline 2-layer MLP)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes).to(self.device)

        # activation
        act = activation.lower()
        if act == "relu":
            self.activation = F.relu
        elif act == "sigmoid":
            self.activation = torch.sigmoid
        elif act == "linear":
            self.activation = lambda z: z
        else:
            raise ValueError(f"Unknown activation '{activation}'. Use 'relu' | 'sigmoid' | 'linear'.")

        # loss + regularization
        self.criterion = nn.CrossEntropyLoss()  # reduction='mean'
        self.reg_parameter = float(reg_parameter)

        # minibatch settings
        self.default_batch_size = int(default_batch_size)
        self.rng = np.random.default_rng(seed)
        self.use_minibatch_for_gradient = bool(use_minibatch_for_gradient)

    # -------- public API --------
    def value(self, point):
        """Return full-batch loss(point) as a numpy scalar."""
        with torch.no_grad():
            loss = self._loss(point)
        return float(loss.detach().cpu().item())

    def gradient(self, point, batch_size=None, idx=None, replace=False):
        """
        By default returns the full-batch gradient.
        If use_minibatch_for_gradient=True, returns a minibatch gradient (respects batch_size/idx).
        """
        if self.use_minibatch_for_gradient:
            return self.stochastic_gradient(point, batch_size=batch_size, idx=idx, replace=replace)

        # Full-batch gradient
        vector_to_parameters(self._to_tensor(point), self._parameters_iter())
        self._zero_grads()
        logits = self._forward(self.X)
        loss = self._apply_loss_and_reg(logits, self.y)
        loss.backward()
        g = parameters_to_vector(
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in self._parameters_iter()
        )
        return g.detach().cpu().numpy().copy()

    def stochastic_gradient(self, point, batch_size=None, idx=None, replace=False):
        """
        Return a minibatch gradient at 'point'.
        - If 'idx' is provided (array-like), use those indices for the batch.
        - Else, sample a random minibatch of size 'batch_size' (defaults to self.default_batch_size).
        """
        if idx is None:
            idx = self._sample_minibatch(batch_size=batch_size, replace=replace)

        vector_to_parameters(self._to_tensor(point), self._parameters_iter())
        self._zero_grads()
        xb = self.X[idx]
        yb = self.y[idx]
        logits = self._forward(xb)
        loss = self._apply_loss_and_reg(logits, yb)
        loss.backward()

        g = parameters_to_vector(
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in self._parameters_iter()
        )
        return g.detach().cpu().numpy().copy()

    def stochastic_gradient_at_points(self, points, batch_size=None, idx=None, replace=False):
        """Minibatch gradient for each point; same batch for all if 'idx' is provided."""
        if idx is None:
            return [self.stochastic_gradient(p, batch_size=batch_size, idx=None, replace=replace) for p in points]
        else:
            return [self.stochastic_gradient(p, batch_size=batch_size, idx=idx, replace=replace) for p in points]

    def dim(self):
        """Number of parameters."""
        with torch.no_grad():
            theta = parameters_to_vector(self._parameters_iter())
        return theta.numel()

    def get_current_point(self):
        """Current parameter vector as numpy array."""
        with torch.no_grad():
            theta = parameters_to_vector(self._parameters_iter()).detach().cpu().numpy().copy()
        return theta

    # -------- internal helpers --------
    def _loss(self, point):
        """Full-batch loss (CE mean + optional std-regularization)."""
        vector_to_parameters(self._to_tensor(point), self._parameters_iter())
        logits = self._forward(self.X)
        return self._apply_loss_and_reg(logits, self.y)

    def _logits(self, point):
        vector_to_parameters(self._to_tensor(point), self._parameters_iter())
        return self._forward(self.X)

    def _check_accuracy(self, point):
        logits = self._logits(point)
        pred = logits.detach().cpu().numpy().argmax(axis=1)
        return float((pred == self.y_np).mean())

    # -------- core NN pieces --------
    def _forward(self, x):
        z1 = self.fc1(x)
        h = self.activation(z1)
        logits = self.fc2(h)
        return logits

    def _apply_loss_and_reg(self, logits, labels):
        """Compute mean CE on provided labels + optional std regularization over all params."""
        loss = self.criterion(logits, labels)
        if self.reg_parameter > 0.0:
            params = torch.cat([p.view(-1) for p in self._parameters_iter()])
            loss = loss + self.reg_parameter * torch.std(params)
        return loss

    def _parameters_iter(self):
        # Fixed order for vectorization
        for p in self.fc1.parameters():
            yield p
        for p in self.fc2.parameters():
            yield p

    def _zero_grads(self):
        for p in self._parameters_iter():
            if p.grad is not None:
                p.grad.zero_()

    def _to_tensor(self, point):
        if isinstance(point, np.ndarray):
            t = torch.from_numpy(point).to(self.device, dtype=torch.float32)
        elif isinstance(point, torch.Tensor):
            t = point.to(self.device, dtype=torch.float32)
        else:
            t = torch.tensor(point, dtype=torch.float32, device=self.device)
        return t

    # -------- minibatch utilities --------
    def _sample_minibatch(self, batch_size=None, replace=False):
        """Return indices for a random minibatch."""
        b = int(batch_size or self.default_batch_size)
        b = min(b, self.N)
        return self.rng.choice(self.N, size=b, replace=replace)
