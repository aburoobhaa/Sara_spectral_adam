import torch

def power_iteration(matrix, num_iters=15, top_k=10):
    n = matrix.shape[0]
    vecs = torch.randn((n, top_k), device=matrix.device)
    vecs, _ = torch.linalg.qr(vecs)

    for _ in range(num_iters):
        vecs = matrix @ vecs
        vecs, _ = torch.linalg.qr(vecs)

    eigenvectors = vecs
    eigenvalues = torch.sum(eigenvectors * (matrix @ eigenvectors), dim=0)
    return eigenvalues, eigenvectors

def spectral_preconditioner(eigenvalues, eigenvectors):
    return torch.sum(eigenvalues.view(-1, 1) * (eigenvectors ** 2), dim=0)
