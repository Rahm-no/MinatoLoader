"""Preprocessing pipelines, expressed as indexed stage lists.

A stage is ``(name, fn)`` where ``fn(payload) -> payload``. The first stage
receives ``{"row": <manifest row>}`` and the last must leave the finished
tensor at ``payload["out"]``. Position in the list is the resume index that
travels with a migrated sample.

Per-sample determinism
----------------------
Each stage seeds the RNG from ``(global_seed, sample_id, stage_index)`` before
running. That matters here for a reason that is easy to miss: a migrated sample
executes its early stages in a foreground worker and its later stages in a
background worker, in a different process with a different RNG stream. Without
per-sample-per-stage seeding, a sample's augmentation would depend on whether
it happened to be migrated, which would make the two loaders produce different
data and invalidate any comparison between them.

Pipeline note (LibriSpeech)
---------------------------
The design brief lists ``Pad -> SpecAugment -> FilterBank -> FrameSplicing ->
normalize``. SpecAugment masks time and frequency bands of a mel spectrogram,
so it cannot run before the filterbank. The order implemented here is
``Pad -> FilterBank -> SpecAugment -> FrameSplicing -> Normalize``, which is
the NeMo/Jasper ordering. Flagged rather than silently changed.
"""

import hashlib
import io
import os

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _sample_seed(global_seed, sample_id, stage_index):
    h = hashlib.blake2b(("%s|%s|%d" % (global_seed, sample_id, stage_index)).encode(),
                        digest_size=8)
    return int.from_bytes(h.digest(), "little") % (2 ** 31 - 1)


def _seeded(stage_index, global_seed, fn):
    """Wrap a stage so its RNG depends only on (seed, sample_id, stage)."""
    import random as _random

    import torch

    def wrapped(payload):
        sid = payload["row"]["sample_id"]
        s = _sample_seed(global_seed, sid, stage_index)
        torch.manual_seed(s)
        _random.seed(s)
        return fn(payload)

    return wrapped


def build_stages(pipeline, kwargs, seed):
    if pipeline not in REGISTRY:
        raise ValueError("unknown pipeline %r; have %s"
                         % (pipeline, sorted(REGISTRY)))
    raw = REGISTRY[pipeline](**(kwargs or {}))
    return [(name, _seeded(i, seed, fn)) for i, (name, fn) in enumerate(raw)]


def stage_names(pipeline, kwargs=None):
    return [name for name, _ in REGISTRY[pipeline](**(kwargs or {}))]


# ---------------------------------------------------------------------------
# OpenImages / any directory of JPEGs
# ---------------------------------------------------------------------------

@register("openimages")
def openimages_pipeline(size=224, jitter=0.4, blur_kernel=9,
                        blur_sigma=(0.1, 2.0), read_to_memory=True):
    """RandomResizedCrop -> HFlip -> ColorJitter -> GaussianBlur -> ToTensor
    -> Normalize, with decode split out as its own stage.

    Decode is deliberately stage 1 (after the file read) rather than fused with
    it, because on this workload decode is where the cost heterogeneity lives
    and a fused read+decode stage would make every migration land at stage 0
    with nothing to resume from.
    """
    from PIL import Image
    from torchvision import transforms as T

    crop = T.RandomResizedCrop(size)
    flip = T.RandomHorizontalFlip(p=0.5)
    jit = T.ColorJitter(brightness=jitter, contrast=jitter,
                        saturation=jitter, hue=min(0.5, jitter / 2.0))
    blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
    to_tensor = T.ToTensor()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def read(payload):
        path = payload["row"]["path"]
        if read_to_memory:
            with open(path, "rb") as fh:
                payload["raw"] = fh.read()
        else:
            payload["raw"] = path
        return payload

    def decode(payload):
        raw = payload.pop("raw")
        src = io.BytesIO(raw) if isinstance(raw, bytes) else raw
        img = Image.open(src)
        img.load()                      # force the actual decode here
        payload["img"] = img.convert("RGB")
        return payload

    def mk(fn, key_in="img", key_out="img"):
        def stage(payload):
            payload[key_out] = fn(payload[key_in])
            return payload
        return stage

    def finish(payload):
        payload["out"] = norm(payload.pop("img"))
        return payload

    return [
        ("read", read),
        ("decode", decode),
        ("random_resized_crop", mk(crop)),
        ("horizontal_flip", mk(flip)),
        ("color_jitter", mk(jit)),
        ("gaussian_blur", mk(blur)),
        ("to_tensor", mk(to_tensor)),
        ("normalize", finish),
    ]


# ---------------------------------------------------------------------------
# OpenImages + SimCLR (two augmented views per sample)
# ---------------------------------------------------------------------------

@register("openimages_simclr")
def openimages_simclr_pipeline(size=224, jitter=0.4, grayscale_p=0.2,
                               blur_kernel=23, blur_sigma=(0.1, 2.0),
                               jitter_p=0.8, read_to_memory=True):
    """SimCLR: the augmentation stack applied twice to the same decoded image.

    The two views are laid out as one *linear* stage list -- read, decode, then
    view 1's six stages, then view 2's six stages, then stack. That keeps the
    resume index meaningful: a migration during view 2 carries the finished
    view 1 with it, and only the interrupted view-2 stage is re-executed. An
    interleaved or parallel two-view implementation would have no single
    integer resume point, which is presumably part of why the original scoping
    excluded SimCLR.

    Two consequences worth measuring rather than assuming:

      * The decoded source image must be retained until view 2 finishes, so
        every checkpoint copy between stages 2 and 13 carries the full decoded
        image *plus* the partial view. On a 2400px OpenImages sample that is
        tens of MB per checkpoint. Compare against --checkpoint_copy false.
      * Per-sample cost roughly doubles while decode cost does not, so decode
        is a smaller share of the total than in the single-view pipeline. That
        should *reduce* migrations-at-stage-0 relative to the single-view runs,
        which is the one respect in which SimCLR suits resume-from-index
        better.
    """
    from PIL import Image
    from torchvision import transforms as T

    base = openimages_simclr_pipeline  # noqa: F841  (marker for introspection)

    crop = T.RandomResizedCrop(size, scale=(0.2, 1.0))
    flip = T.RandomHorizontalFlip(p=0.5)
    jit = T.RandomApply([T.ColorJitter(jitter, jitter, jitter,
                                       min(0.5, jitter / 2.0))], p=jitter_p)
    gray = T.RandomGrayscale(p=grayscale_p)
    blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
    to_tensor = T.ToTensor()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def read(payload):
        path = payload["row"]["path"]
        if read_to_memory:
            with open(path, "rb") as fh:
                payload["raw"] = fh.read()
        else:
            payload["raw"] = path
        return payload

    def decode(payload):
        raw = payload.pop("raw")
        src = io.BytesIO(raw) if isinstance(raw, bytes) else raw
        img = Image.open(src)
        img.load()
        payload["src"] = img.convert("RGB")
        return payload

    def view_stages(tag):
        key = "v%s" % tag

        def start(payload):
            # view 2 must re-crop from the decoded source, so `src` stays live
            payload[key] = crop(payload["src"])
            return payload

        def step(fn):
            def stage(payload):
                payload[key] = fn(payload[key])
                return payload
            return stage

        return [
            ("v%s_random_resized_crop" % tag, start),
            ("v%s_horizontal_flip" % tag, step(flip)),
            ("v%s_color_jitter" % tag, step(jit)),
            ("v%s_grayscale" % tag, step(gray)),
            ("v%s_gaussian_blur" % tag, step(blur)),
            ("v%s_to_tensor" % tag, step(to_tensor)),
            ("v%s_normalize" % tag, step(norm)),
        ]

    def stack(payload):
        import torch
        payload["out"] = torch.stack([payload.pop("v1"), payload.pop("v2")])
        payload.pop("src", None)
        return payload

    return ([("read", read), ("decode", decode)]
            + view_stages("1") + view_stages("2")
            + [("stack_views", stack)])


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------

@register("librispeech")
def librispeech_pipeline(sample_rate=16000, n_mels=64, win_ms=20.0, hop_ms=10.0,
                         pad_to_s=17.0, freq_mask=8, time_mask=25,
                         splice_frames=3):
    import torch
    import torchaudio
    from torchaudio import transforms as AT

    win = int(sample_rate * win_ms / 1000.0)
    hop = int(sample_rate * hop_ms / 1000.0)
    pad_samples = int(sample_rate * pad_to_s)

    fbank = AT.MelSpectrogram(sample_rate=sample_rate, n_fft=max(512, win),
                              win_length=win, hop_length=hop, n_mels=n_mels)
    fmask = AT.FrequencyMasking(freq_mask_param=freq_mask)
    tmask = AT.TimeMasking(time_mask_param=time_mask)

    def load(payload):
        wav, sr = torchaudio.load(payload["row"]["path"])
        if sr != sample_rate:
            wav = AT.Resample(sr, sample_rate)(wav)
        payload["wav"] = wav.mean(dim=0, keepdim=True)
        return payload

    def pad(payload):
        wav = payload["wav"]
        n = wav.shape[-1]
        if n < pad_samples:
            wav = torch.nn.functional.pad(wav, (0, pad_samples - n))
        else:
            wav = wav[..., :pad_samples]
        payload["wav"] = wav
        return payload

    def filterbank(payload):
        payload["feat"] = (fbank(payload.pop("wav")) + 1e-6).log()
        return payload

    def specaugment(payload):
        payload["feat"] = tmask(fmask(payload["feat"]))
        return payload

    def frame_splice(payload):
        feat = payload["feat"]
        t = feat.shape[-1] - (feat.shape[-1] % splice_frames)
        feat = feat[..., :t]
        c, m, t = feat.shape
        payload["feat"] = feat.reshape(c, m * splice_frames, t // splice_frames)
        return payload

    def normalize(payload):
        feat = payload.pop("feat")
        payload["out"] = (feat - feat.mean()) / (feat.std() + 1e-5)
        return payload

    return [
        ("load_audio", load),
        ("pad", pad),
        ("filterbank", filterbank),
        ("specaugment", specaugment),
        ("frame_splice", frame_splice),
        ("normalize", normalize),
    ]


# ---------------------------------------------------------------------------
# synthetic image corpus -- lets the Mac smoke test run with no dataset
# ---------------------------------------------------------------------------

@register("synthetic_simclr")
def synthetic_simclr_pipeline(**kwargs):
    """SimCLR stack over the synthetic corpus, for the smoke test."""
    return openimages_simclr_pipeline(**kwargs)


@register("synthetic_image")
def synthetic_image_pipeline(**kwargs):
    """Identical to the OpenImages pipeline. The heterogeneity comes from the
    corpus generator writing JPEGs of widely varying dimensions, so decode cost
    varies the way it does on real OpenImages."""
    return openimages_pipeline(**kwargs)


def make_synthetic_corpus(out_dir, n, seed=0, min_side=128, max_side=2400,
                          heavy_frac=0.15, quality=92):
    """Write real JPEGs with a heavy-tailed size distribution."""
    import random as _random

    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = _random.Random(seed)
    nprng = np.random.RandomState(seed)
    paths = []
    for i in range(n):
        if rng.random() < heavy_frac:
            side = rng.randint(int(max_side * 0.6), max_side)
        else:
            side = rng.randint(min_side, int(min_side * 3))
        h = int(side * rng.uniform(0.6, 1.0))
        arr = nprng.randint(0, 256, size=(h, side, 3), dtype=np.uint8)
        path = os.path.join(out_dir, "syn_%06d.jpg" % i)
        Image.fromarray(arr).save(path, quality=quality)
        paths.append(path)
    return paths
