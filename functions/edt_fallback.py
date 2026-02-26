# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Modified by RotoForge AI to support non-CUDA platforms (macOS MPS / CPU).

# pyre-unsafe

"""Euclidean distance transform (EDT) — with Triton GPU kernel + PyTorch fallback."""

import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# -----------------------------------------------------------------------
# PyTorch-only fallback (works on CPU, MPS, and CUDA)
# -----------------------------------------------------------------------

def _edt_1d_felzenszwalb(f: torch.Tensor) -> torch.Tensor:
    """1-D EDT squared on the last dimension using Felzenszwalb & Huttenlocher."""
    n = f.shape[-1]
    device = f.device
    flat = f.reshape(-1, n)
    B = flat.shape[0]

    d = torch.full_like(flat, float("inf"))
    v = torch.zeros(B, n, dtype=torch.long, device=device)
    z = torch.full((B, n + 1), float("inf"), dtype=flat.dtype, device=device)
    z[:, 0] = float("-inf")

    k = torch.zeros(B, dtype=torch.long, device=device)

    for q in range(1, n):
        fq = flat[:, q]
        while True:
            idx_k = k.clamp(min=0)
            vk = v[torch.arange(B, device=device), idx_k]
            fvk = flat[torch.arange(B, device=device), vk]
            s = ((fq - fvk) + (q * q - vk.float() * vk.float())) / (2.0 * (q - vk.float()))
            zk = z[torch.arange(B, device=device), idx_k]
            can_pop = (s <= zk) & (k > 0)
            if not can_pop.any():
                break
            k = k - can_pop.long()

        k = k + 1
        v[torch.arange(B, device=device), k] = q
        idx_k = k
        fvk_cur = flat[torch.arange(B, device=device), v[torch.arange(B, device=device), idx_k - 1]]
        vk_cur = v[torch.arange(B, device=device), idx_k - 1].float()
        s_final = ((fq - fvk_cur) + (q * q - vk_cur * vk_cur)) / (2.0 * (q - vk_cur))
        z[torch.arange(B, device=device), idx_k] = s_final
        safe = (idx_k + 1 < n).long()
        z[torch.arange(B, device=device), (idx_k + 1).clamp(max=n)] = torch.where(
            safe.bool(), torch.tensor(float("inf"), device=device), z[torch.arange(B, device=device), (idx_k + 1).clamp(max=n)]
        )

    cur_k = torch.zeros(B, dtype=torch.long, device=device)
    for q in range(n):
        while True:
            next_k = (cur_k + 1).clamp(max=n - 1)
            z_next = z[torch.arange(B, device=device), next_k + 1]
            advance = (cur_k + 1 < n) & (z_next < q)
            if not advance.any():
                break
            cur_k = cur_k + advance.long()
        r = v[torch.arange(B, device=device), cur_k]
        old_val = flat[torch.arange(B, device=device), r]
        d[:, q] = old_val + (q - r.float()) ** 2

    return d.reshape(f.shape)


def _edt_pytorch(data: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch EDT — works on any device. Expects (B, H, W) bool/int."""
    f = torch.where(data.bool(), 1e18, 0.0)
    # Horizontal pass
    f = _edt_1d_felzenszwalb(f)
    # Vertical pass — transpose, run 1-D, transpose back
    f = f.permute(0, 2, 1)
    f = _edt_1d_felzenszwalb(f)
    f = f.permute(0, 2, 1)
    return f.sqrt()


# -----------------------------------------------------------------------
# Triton kernel (original Meta implementation, CUDA only)
# -----------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape containing the EDT
        (equivalent to batched cv2.distanceTransform with DIST_L2).
    """
    assert data.dim() == 3

    if not data.is_cuda or not _HAS_TRITON:
        return _edt_pytorch(data)

    B, H, W = data.shape
    data = data.contiguous()

    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, H)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=True,
    )

    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=False,
    )
    return output.sqrt()
