"""Differentiable raymarching in PyTorch: reconstruct a defect from 3 realistic radiographs + 2D masks.

Pipeline (digital-twin comparison, standard for industrial parts with a known CAD model):
  measured radiograph ──linearize──> material path length L_meas
  simulated defect-free radiograph (same physics) ──linearize──> L_ref
  ΔL = L_ref - L_meas = chord length through the void, per pixel      <- tomographic data
  backprojected 2D masks -> visual hull                                 <- support constraint
  find a void volume v (0..1, inside the support) whose projections match ΔL, by gradient descent.

Run:  python torch_recon.py   (imports raymarch.py; writes torch_recon.png)
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

import raymarch as rm

dev = "cuda" if torch.cuda.is_available() else "cpu"


def ray_samples(angle):
    """Geometry is fixed, so precompute every ray's sample points once. valid=False past the exit point."""
    o, d = rm.camera_rays(angle)
    tnear, tfar = rm.intersect_box(o, d)
    k = np.arange(int(np.ceil((tfar - tnear).max() / rm.STEP)))
    t = tnear[:, None] + (k + 0.5) * rm.STEP  # (rays, steps), same midpoint rule as rm.march
    p = o[:, None] + t[..., None] * d[:, None]
    return (torch.tensor(p, dtype=torch.float32, device=dev),
            torch.tensor(t < tfar[:, None], dtype=torch.float32, device=dev))


def project(vol, pts, valid):
    """Differentiable line integral: trilinear sampling + Riemann sum, i.e. rm.line_integral as tensor ops.
    grid_sample wants a (1,1,D,H,W) volume and (x,y,z) coords indexing (W,H,D), so store our [x,y,z] volume as [z,y,x]."""
    s = F.grid_sample(vol.permute(2, 1, 0)[None, None], pts[None, :, :, None], align_corners=True)
    return (s.view(valid.shape) * valid).sum(-1).view(rm.DET_RES, rm.DET_RES) * rm.STEP


def tv(v):
    """Anisotropic total variation: favours piecewise-constant volumes (a void is either there or not)."""
    return sum((v.diff(dim=i)).abs().mean() for i in range(3))


if __name__ == "__main__":
    angles = np.deg2rad([0, 60, 120])
    cad, part, defect = rm.make_phantom()
    E, w = rm.spectrum()
    mu = rm.mu_mm("Al", 2.699, E)
    rng = np.random.default_rng(0)
    geo = [ray_samples(a) for a in angles]

    # The torch projector must agree with the numpy raymarcher, otherwise the gradients optimize the wrong model.
    diff = (project(torch.tensor(cad, device=dev), *geo[0]).cpu().numpy() - rm.line_integral(cad, angles[0])).max()
    assert abs(diff) < 1e-3, f"torch and numpy projectors disagree by {diff}"

    dL, masks = [], []
    for a in angles:
        meas = rm.linearize(rm.radiograph(part, a, w, mu, rng=rng)[0], w, mu)
        ref = rm.linearize(rm.radiograph(cad, a, w, mu, photons=None)[0], w, mu)  # same physics, no defect, no noise
        dL.append(torch.tensor((ref - meas) / rm.MM, dtype=torch.float32, device=dev))  # world units
        # ponytail: masks come from projecting the ground-truth defect; swap in your 2D segmentation model's output.
        masks.append(rm.line_integral(defect.astype(np.float32), a) > 0.02)

    hull = sum(rm.backproject(m, a).astype(int) for m, a in zip(masks, angles)) == len(angles)
    support = torch.tensor(binary_dilation(hull, iterations=1) & (cad > 0), dtype=torch.float32, device=dev)

    logits = torch.zeros((rm.N,) * 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    losses = []
    for it in range(400):
        v = torch.sigmoid(logits) * support  # void fraction in [0,1], zero outside the hull
        data = sum(F.mse_loss(project(v, *g), y) for g, y in zip(geo, dL))
        loss = data + 2e-4 * tv(v)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(data.item())

    v = (torch.sigmoid(logits) * support).detach().cpu().numpy()
    rec = v > 0.5
    iou = lambda a: (a & defect).sum() / (a | defect).sum()
    world = lambda ijk: ijk / (rm.N - 1) * 2 - 1
    cerr = lambda a: np.linalg.norm(world(np.argwhere(a).mean(0)) - world(np.argwhere(defect).mean(0))) * rm.MM
    print(f"device {dev}, final data loss {losses[-1]:.2e}")
    print(f"visual hull : {hull.sum():4d} voxels  IoU {iou(hull):.2f}  centre error {cerr(hull):.2f} mm")
    print(f"optimized   : {rec.sum():4d} voxels  IoU {iou(rec):.2f}  centre error {cerr(rec):.2f} mm  (truth {defect.sum()})")
    assert iou(rec) > iou(hull), "gradient-based reconstruction should beat the visual hull"

    zi = int(np.argwhere(defect)[:, 2].mean())
    xs = slice(int(np.argwhere(defect)[:, 0].mean()) - 10, int(np.argwhere(defect)[:, 0].mean()) + 10)
    ys = slice(int(np.argwhere(defect)[:, 1].mean()) - 10, int(np.argwhere(defect)[:, 1].mean()) + 10)
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for k in range(3):
        y, p = dL[k].cpu().numpy(), project(torch.tensor(v, device=dev), *geo[k]).cpu().numpy()
        ax[0, k].imshow(np.hstack([y, p]) * rm.MM, origin="lower", cmap="viridis", vmin=0)
        ax[0, k].set_title(f"{np.rad2deg(angles[k]):.0f}°: measured ΔL | reprojected (mm)"); ax[0, k].axis("off")
    ax[0, 3].semilogy(losses); ax[0, 3].set(title="data loss", xlabel="Adam iteration")
    for a, (vol, title) in zip(ax[1], [(defect, "ground truth"), (hull, "visual hull (masks only)"),
                                       (v, "optimized void fraction"), (rec, "optimized > 0.5")]):
        a.imshow(vol[xs, ys, zi].T.astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{title}, z slice {zi}"); a.axis("off")
    fig.tight_layout(); fig.savefig("torch_recon.png", dpi=100)
    plt.show()
