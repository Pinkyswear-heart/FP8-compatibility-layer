from dataclasses import dataclass, field

PROFILES = ("llm_safe", "image_safe")  # the only profiles v1 commits to


@dataclass
class FP8Config:
    fmt: str = "e4m3fn"
    profile: str = "llm_safe"
    weight_group: int | None = None     # None = per output channel; 128 = per-K-group
    activation: str = "fp16"            # W8A16 only in v1
    kv_cache: str = "fp16"
    fallback: str = "tbf16"             # tbf16 | fp16 | native
    min_features: int | None = None     # None -> per-profile default (llm 1024, image 256)
    pow2_scale: bool = True
    large_m: int = 256                  # M at/above which the large-M backend is used
    large_m_backend: str = "chunked"    # fused (WMMA kernel) | chunked (fixed-workspace dequant + cuBLAS)
    workspace_mb: int = 32              # fixed workspace for the chunked backend
    nan_check: bool = True              # fall back a module (and log it) on NaN/Inf output
    extra_skip: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"profile {self.profile!r} not supported in v1; use one of {PROFILES}")
        if self.activation != "fp16":
            raise NotImplementedError("W8A8 is a later phase; v1 is W8A16 only")
        if self.kv_cache != "fp16":
            raise NotImplementedError("FP8 KV cache is a later phase")
        if self.large_m_backend not in ("fused", "chunked"):
            raise ValueError("large_m_backend must be 'fused' or 'chunked'")
        if self.min_features is None:
            # LLM projections are wide; DiT/UNet attention and MLP layers are often only 256-1024
            # wide and still worth quantizing, but adaLN/time/final modules are excluded by name.
            self.min_features = 1024 if self.profile == "llm_safe" else 256
