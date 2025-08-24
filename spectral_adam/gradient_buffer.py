import torch

class GradientBuffer:
    def __init__(self, buffer_size, param_size, device='cpu'):
        self.W = buffer_size
        self.device = device
        self.buffer = torch.zeros((buffer_size, param_size), device=device)
        self.index = 0
        self.full = False

    def add(self, grad_vector):
        self.buffer[self.index] = grad_vector
        self.index = (self.index + 1) % self.W
        if self.index == 0:
            self.full = True

    def get_matrix(self):
        return self.buffer if self.full else self.buffer[:self.index]

    def is_ready(self, top_k):
        return self.full or self.index >= top_k
