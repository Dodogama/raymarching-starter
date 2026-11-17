"""Raymarching demo for X-ray style imaging.

1. Forward:  march rays from an X-ray source through a voxel volume -> synthetic radiographs, with the
             physics that make them realistic: polychromatic beam, focal spot, scatter, detector blur, noise.
2. Backward: march rays from 2D segmentation-mask pixels back into the volume -> 3D defect location
             from a sparse set of views (visual hull / space carving).

Run:  python raymarch.py   (needs numpy, scipy, matplotlib, spekpy, xraylib)
      writes physics_demo.png and raymarch_demo.png
"""
import numpy as np
import matplotlib.pyplot as plt
import spekpy
import xraylib
from scipy.ndimage import gaussian_filter

N = 64                         # voxels per side; the volume fills the world box [-1, 1]^3
DET_RES = 128                  # detector pixels per side
DET_SIZE = 2.6                 # detector side length (world units)
SRC_DIST, DET_DIST = 3.0, 1.5  # source / detector distance from the rotation axis
STEP = 1.0 / N                 # march step = half a voxel (voxel width = 2/N)
MM = 25.0                      # mm per world unit: 50 mm field of view, 30 mm diameter part


def make_phantom():
    """Aluminium cylinder (the CAD model) with a spherical internal void (porosity defect)."""
    g = np.linspace(-1, 1, N)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    cad = (x**2 + y**2 < 0.6**2) & (np.abs(z) < 0.8)
    defect = (x - 0.25)**2 + (y + 0.15)**2 + (z - 0.3)**2 < 0.12**2
    f = lambda v: v.astype(np.float32)
    return f(cad), f(cad * ~defect), defect


def camera_rays(angle, src_shift=(0.0, 0.0)):
    """Cone-beam geometry: one point source, one ray per detector pixel, gantry rotates about z.
    src_shift moves the source within its plane (used to model a finite focal spot)."""
    c, s = np.cos(angle), np.sin(angle)
    fwd, right, up = np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0.0, 0.0, 1.0])
    src = -SRC_DIST * fwd + src_shift[0] * right + src_shift[1] * up
    u = (np.arange(DET_RES) + 0.5) / DET_RES * DET_SIZE - DET_SIZE / 2
    uu, vv = np.meshgrid(u, u, indexing="xy")  # image rows = up axis, cols = right axis
    pix = DET_DIST * fwd + uu[..., None] * right + vv[..., None] * up
    d = pix - src
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.broadcast_to(src, d.shape).reshape(-1, 3), d.reshape(-1, 3)


def intersect_box(o, d):
    """Slab method: distance where each ray enters/exits [-1, 1]^3. Misses have tfar < tnear."""
    with np.errstate(divide="ignore"):
        inv = 1.0 / d
    t0, t1 = (-1 - o) * inv, (1 - o) * inv
    return np.maximum(np.minimum(t0, t1).max(-1), 0), np.maximum(t0, t1).min(-1)


def sample(vol, p):
    """Trilinear interpolation of vol at world points p (M, 3)."""
    g = (p + 1) / 2 * (N - 1)
    i = np.clip(np.floor(g).astype(int), 0, N - 2)
    f = g - i
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * vol[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return out


def march(o, d):
    """The raymarch loop. Every ray advances STEP per iteration; yields sample points of rays still inside."""
    tnear, tfar = intersect_box(o, d)
    t = tnear + STEP / 2  # midpoint rule
    while (alive := t < tfar).any():
        yield o[alive] + t[alive, None] * d[alive], alive
        t += STEP


def line_integral(vol, angle, src_shift=(0.0, 0.0)):
    """∫vol ds along every detector ray (world units). vol = μ gives -log(I/I0) (Beer-Lambert, one energy);
    vol = material mask gives the path length through that material, which the polychromatic model needs."""
    o, d = camera_rays(angle, src_shift)
    acc = np.zeros(len(o))
    for p, alive in march(o, d):
        acc[alive] += sample(vol, p) * STEP
    return acc.reshape(DET_RES, DET_RES)


def emission(q, mu, angle):
    """Emission imaging with self-absorption (SPECT, X-ray fluorescence). Marching toward the detector, each
    step first attenuates everything emitted further back, then adds its own emission ('over' compositing)."""
    o, d = camera_rays(angle)
    acc = np.zeros(len(o))
    for p, alive in march(o, d):
        acc[alive] = acc[alive] * np.exp(-sample(mu, p) * STEP) + sample(q, p) * STEP
    return acc.reshape(DET_RES, DET_RES)


def backproject(mask, angle):
    """Mark every voxel that a ray through a masked pixel passes through (the defect's cone for this view)."""
    o, d = camera_rays(angle)
    m = mask.ravel()
    hit = np.zeros((N,) * 3, bool)
    for p, _ in march(o[m], d[m]):
        idx = np.clip(np.rint((p + 1) / 2 * (N - 1)).astype(int), 0, N - 1)
        hit[tuple(idx.T)] = True
    return hit


# --- X-ray physics ----------------------------------------------------------------------------------------

def spectrum(kvp=120, al_filter_mm=2.0):
    """Tungsten tube spectrum from SpekPy, weighted by photon energy (energy-integrating detector), normalized."""
    s = spekpy.Spek(kvp=kvp, th=12, dk=1)
    s.filter("Al", al_filter_mm)
    E, phi = s.get_spectrum()
    w = phi * E
    return E, w / w.sum()


def mu_mm(compound, density, E):
    """Linear attenuation coefficient (1/mm) at energies E (keV) from xraylib total cross sections."""
    return np.array([xraylib.CS_Total_CP(compound, float(e)) for e in E]) * density / 10


def polychromatic(L_mm, w, mu):
    """I/I0 = Σ_E w(E)·exp(-μ(E)·L). Low energies are absorbed first, so the beam 'hardens' with depth
    and -log(I/I0) grows slower than linearly in L (cupping artifacts in CT)."""
    return np.exp(-np.multiply.outer(L_mm, mu)) @ w


def radiograph(material, angle, w, mu, focal_mm=0.8, scatter_amp=0.15, blur_px=0.8, photons=2e4, rng=None):
    """Realistic flat-field-corrected projection of a single-material part.
    Returns (image, primary, scatter). photons=None skips noise (e.g. for a simulated reference)."""
    r = focal_mm / MM / 2
    shifts = [(sx, sy) for sx in (-r, r) for sy in (-r, r)]  # finite focal spot -> penumbra at edges
    primary = np.mean([polychromatic(line_integral(material, angle, s) * MM, w, mu) for s in shifts], axis=0)
    # ponytail: scatter-kernel approximation (broad blur of the single-scatter-like term I·τ); use Monte Carlo
    # (GATE / MC-GPU) or a learned scatter estimator when small defects or high scatter-to-primary ratios matter.
    scatter = scatter_amp * gaussian_filter(primary * -np.log(primary), sigma=DET_RES / 8)
    img = primary + scatter
    if photons:
        img = (rng or np.random.default_rng(0)).poisson(img * photons) / photons  # quantum noise
    img = gaussian_filter(img, blur_px)  # scintillator light spread: blurs (and correlates) the noise
    return np.clip(img, 1 / (photons or 1e6), None), primary, scatter


def linearize(img, w, mu, L_max_mm=100):
    """Single-material beam-hardening correction: invert the monotone I(L) curve with a lookup table -> path length (mm)."""
    L = np.linspace(0, L_max_mm, 2000)
    return np.interp(-np.log(img), -np.log(polychromatic(L, w, mu)), L)


if __name__ == "__main__":
    angles = np.deg2rad([0, 60, 120])  # sparse: only 3 views
    cad, part, defect = make_phantom()
    E, w = spectrum()
    mu = mu_mm("Al", 2.699, E)
    mu_eff = mu @ w  # what a monochromatic model would use
    rng = np.random.default_rng(0)

    # ---- physics walkthrough, view 0 --------------------------------------------------------------------
    L_true = line_integral(part, 0.0) * MM
    assert np.abs(linearize(polychromatic(L_true, w, mu), w, mu) - L_true).max() < 0.1, "LUT inversion broken"
    img, primary, scatter = radiograph(part, 0.0, w, mu, rng=rng)
    c = DET_RES // 2
    print(f"spectrum mean energy {E @ w:.1f} keV, μ_eff {mu_eff:.4f}/mm, "
          f"scatter-to-primary at centre {scatter[c, c] / primary[c, c]:.2f}")

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    panels = [(mu_eff * L_true, "monochromatic -log(I/I0)"),
              (-np.log(polychromatic(L_true, w, mu)), "polychromatic (point source)"),
              (-np.log(primary + scatter), "+ focal spot + scatter"),
              (-np.log(img), "+ Poisson noise + detector blur")]
    for a, (im, title) in zip(ax[0], panels):
        a.imshow(im, cmap="gray", origin="lower"); a.set_title(title); a.axis("off")

    a = ax[1, 0]
    a.plot(L_true[c], "k", label="true path length")
    a.plot(-np.log(polychromatic(L_true, w, mu))[c] / mu_eff, label="poly, read as mono (beam hardening)")
    a.plot(linearize(primary + scatter, w, mu)[c], label="linearized, with scatter")
    a.plot(linearize(img, w, mu)[c], lw=0.7, alpha=0.7, label="linearized, realistic")
    a.set(title="centre row: path length estimate", xlabel="detector column", ylabel="mm"); a.legend(fontsize=7)

    a = ax[1, 1]
    a.imshow(scatter / primary, origin="lower"); a.set_title("scatter / primary"); a.axis("off")

    mu_w = mu_eff * MM * part  # attenuation per world unit, for the emission example
    for a, (im, title) in zip(ax[1, 2:], [(line_integral(part, 0.0), "emission, no self-absorption"),
                                          (emission(part, mu_w, 0.0), "emission, with self-absorption")]):
        a.imshow(im, cmap="magma", origin="lower"); a.set_title(title); a.axis("off")
    fig.tight_layout(); fig.savefig("physics_demo.png", dpi=100)

    # ---- sparse-view localization ------------------------------------------------------------------------
    imgs = [radiograph(part, a, w, mu, rng=rng)[0] for a in angles]
    # ponytail: masks come from projecting the ground-truth defect; swap in your 2D segmentation model's output.
    masks = [line_integral(defect.astype(np.float32), a) > 0.02 for a in angles]

    votes = sum(backproject(m, a).astype(int) for m, a in zip(masks, angles))
    hull = votes == len(angles)  # voxel seen as "defect" by every view

    world = lambda ijk: ijk / (N - 1) * 2 - 1
    est, true = world(np.argwhere(hull).mean(0)), world(np.argwhere(defect).mean(0))
    err = np.linalg.norm(est - true)
    print(f"true defect centre {true.round(3)}  estimated {est.round(3)}  error {err * MM:.2f} mm")
    print(f"hull voxels {hull.sum()} vs true defect voxels {defect.sum()}")
    assert err < 0.05, "localization error too large"

    fig = plt.figure(figsize=(16, 8))
    for k, a in enumerate(angles):
        for row, title in [(0, "radiograph"), (1, "2D mask")]:
            ax = fig.add_subplot(2, 4, k + 1 + 4 * row)
            ax.imshow(-np.log(imgs[k]), cmap="gray", origin="lower")  # -log(I/I0) ≈ ∫mu ds
            if row:
                ax.contour(masks[k], [0.5], colors="r")
            ax.set_title(f"{title} {np.rad2deg(a):.0f}°"); ax.axis("off")

    zi = int(np.argwhere(defect)[:, 2].mean())
    ax = fig.add_subplot(2, 4, 4)
    im = ax.imshow(votes[:, :, zi].T, origin="lower", extent=[-1, 1, -1, 1])
    fig.colorbar(im, ax=ax, label="# views voting defect")
    ax.set_title(f"backprojected votes, z slice {zi}")

    ax = fig.add_subplot(2, 4, 8, projection="3d")
    for vox, col, lab in [(hull, "r", "reconstructed (hull)"), (defect, "b", "ground truth")]:
        p = world(np.argwhere(vox))
        ax.scatter(*p.T, s=4, c=col, alpha=0.3, label=lab)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title="3D localization")
    ax.legend()

    fig.tight_layout()
    fig.savefig("raymarch_demo.png", dpi=100)
    plt.show()
