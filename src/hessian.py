import torch
from pyhessian import hessian as Hessian


def _make_hessian(model, criterion, loader, device):
    model.eval()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    return Hessian(model, criterion, data=(inputs, targets), cuda=(device.type == 'cuda'))


def _unwrap_eigenvalue(val):
    if isinstance(val, (list, tuple)):
        val = val[0]
    return float(val)


def stable_rank(lambda_max, trace):
    if lambda_max == 0.0:
        return float('inf')
    return trace / lambda_max


def compute_all(model, criterion, loader, device, num_iters=100, num_samples=30):
    """Compute lambda_max, trace, and stable_rank with a single Hessian object."""
    h = _make_hessian(model, criterion, loader, device)

    eigenvalues, _ = h.eigenvalues(maxIter=num_iters, tol=1e-3, top_n=1)
    estimates = h.trace(maxIter=num_samples, tol=1e-3)

    lmax = _unwrap_eigenvalue(eigenvalues[0])
    tr   = float(sum(estimates) / len(estimates))

    return lmax, tr, stable_rank(lmax, tr)
