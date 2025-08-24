import torch
from torch.optim.optimizer import Optimizer
from gradient_buffer import GradientBuffer
from spectral_utils import power_iteration, spectral_preconditioner

class SpectralAdam(Optimizer):
    def __init__(self, params, lr=0.01,betas=(0.9, 0.99),
                 eps=1e-7, buffer_size=100, K=3, top_k=50, device='cpu', proj_dim=512):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        self.K = K
        self.top_k = top_k
        self.step_num = 0
        self.device = device
        self.proj_dim = proj_dim

        self._params = [p for group in self.param_groups for p in group['params']]
        self.total_size = sum(p.numel() for p in self._params)

        # Random projection matrix and buffer on optimizer/device
        self.P = torch.randn(self.total_size, self.proj_dim, device=self.device)
        self.P = torch.nn.functional.normalize(self.P, dim=0)

        self.buffer = GradientBuffer(buffer_size, self.proj_dim, device=self.device)
        self.v_t = torch.zeros(self.total_size, device=self.device)
        self.m_t = torch.zeros(self.total_size, device=self.device)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        grad_vector = torch.cat([p.grad.view(-1) for p in self._params])
        grad_proj = grad_vector @ self.P  # shape: (proj_dim,)
        self.buffer.add(grad_proj)

        beta1, beta2 = self.param_groups[0]['betas']
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']

        self.step_num += 1

        # Update second moment estimate using spectral info if needed
        if self.step_num % self.K == 0 and self.buffer.is_ready(self.top_k):
            G = self.buffer.get_matrix()  # shape: (buffer_size, proj_dim) on device
            C = (G.T @ G) / (G.shape[0] - 1)  # shape: (proj_dim, proj_dim)
            λ, V = power_iteration(C, top_k=self.top_k)
            # Project eigenvectors back to parameter space and transpose to (top_k, total_size)
            V_param = (self.P @ V).t()  # shape: (top_k, total_size)
            s_t = spectral_preconditioner(λ, V_param)
            self.v_t = beta2 * self.v_t + (1 - beta2) * s_t
        else:
            self.v_t = beta2 * self.v_t + (1 - beta2) * grad_vector ** 2

        # Update first moment
        self.m_t = beta1 * self.m_t + (1 - beta1) * grad_vector

        # Bias correction
        m_hat = self.m_t / (1 - beta1 ** self.step_num)
        v_hat = self.v_t / (1 - beta2 ** self.step_num)

        update = -lr * m_hat / (v_hat.sqrt() + eps)

        # Apply update
        offset = 0
        for p in self._params:
            sz = p.numel()
            p.data.add_(update[offset:offset + sz].view_as(p))
            offset += sz

        return loss
