import torch


def _ensure_mask(pair_mask, z):
    # pair_mask: [B,N,N] or [B,N,N,1]
    if pair_mask.dim() == 3:
        pair_mask = pair_mask.unsqueeze(-1)
    pair_mask = pair_mask.to(dtype=z.dtype, device=z.device)
    pair_mask = torch.maximum(pair_mask, pair_mask.transpose(1, 2))
    return pair_mask


def _masked_symmetrize(x, M):
    x = x * M
    return 0.5 * (x + x.transpose(1, 2)) * M


def _make_block_mask(M, block_radius=2):
    # M: [B,N,N,1]
    B, N, _, _ = M.shape
    block = torch.zeros_like(M)
    active = (M[..., 0] > 0)

    for b in range(B):
        idx = torch.nonzero(active[b], as_tuple=False)
        if idx.numel() == 0:
            continue
        k = torch.randint(0, idx.shape[0], (1,), device=M.device).item()
        i0, j0 = idx[k].tolist()

        i1 = max(0, i0 - block_radius)
        i2 = min(N, i0 + block_radius + 1)
        j1 = max(0, j0 - block_radius)
        j2 = min(N, j0 + block_radius + 1)

        block[b, i1:i2, j1:j2, 0] = 1.0

    return block * M


def _default_fast_score(z_cur, M):
    return torch.zeros(z_cur.shape[0], device=z_cur.device, dtype=z_cur.dtype)


def _energy(delta, z_base, M, lambda_anchor=1.0, lambda_fast=0.0, fast_score_fn=None):
    # delta: [B,N,N,C]
    delta = delta * M
    delta = _masked_symmetrize(delta, M)
    z_cur = z_base + delta

    e_anchor = (delta ** 2).sum(dim=(1, 2, 3))
    if fast_score_fn is None:
        e_fast = _default_fast_score(z_cur, M)
    else:
        e_fast = fast_score_fn(z_cur, M)  # should return [B]

    e = lambda_anchor * e_anchor + lambda_fast * e_fast
    return e, z_cur, delta


@torch.no_grad()
def sample_interface_z_mcmc(
    z_base,                 # [B,N,N,C]
    pair_mask,              # [B,N,N] or [B,N,N,1]
    n_steps=100,
    burn_in=20,
    thin=5,
    step_scale=0.03,
    block_radius=2,
    lambda_anchor=1.0,
    lambda_fast=0.0,
    fast_score_fn=None,     
    clamp_delta=2.0,
):
    """
    block Gaussian MH
    """
    M = _ensure_mask(pair_mask, z_base)

    delta = torch.zeros_like(z_base)

    e_cur, z_cur, delta = _energy(
        delta, z_base, M,
        lambda_anchor=lambda_anchor,
        lambda_fast=lambda_fast,
        fast_score_fn=fast_score_fn,
    )

    accepted = torch.zeros(z_base.shape[0], device=z_base.device)
    samples = []
    trace_e = []
    samples.append(z_cur.clone())

    for t in range(n_steps):
        block = _make_block_mask(M, block_radius=block_radius)
        noise = torch.randn_like(delta) * step_scale

        delta_new = delta + block * noise
        if clamp_delta is not None:
            delta_new = delta_new.clamp(-clamp_delta, clamp_delta)

        e_new, z_new, delta_new = _energy(
            delta_new, z_base, M,
            lambda_anchor=lambda_anchor,
            lambda_fast=lambda_fast,
            fast_score_fn=fast_score_fn,
        )

        log_alpha = -(e_new - e_cur)
        logu = torch.log(torch.rand_like(log_alpha).clamp_min(1e-12))
        accept = (logu < log_alpha)  # [B]

        view = accept.view(-1, 1, 1, 1)
        delta = torch.where(view, delta_new, delta)
        z_cur = torch.where(view, z_new, z_cur)
        e_cur = torch.where(accept, e_new, e_cur)

        accepted += accept.float()
        trace_e.append(e_cur.clone())

        if t >= burn_in and ((t - burn_in) % thin == 0):
            samples.append(z_cur.clone())

    if len(samples) > 0:
        samples = torch.stack(samples, dim=0)   # [S,B,N,N,C]
    else:
        samples = torch.empty(
            0, *z_base.shape,
            device=z_base.device,
            dtype=z_base.dtype
        )

    trace_e = torch.stack(trace_e, dim=0)       # [T,B]
    accept_rate = accepted / float(n_steps)

    return {
        "samples_z": samples,
        "final_z": z_cur,
        "final_delta": delta,
        "accept_rate": accept_rate,
        "trace_energy": trace_e,
    }