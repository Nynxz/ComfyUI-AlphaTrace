import copy

import torch

import comfy.samplers
import nodes
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced


class AlphaTracer:
    """Records the denoised prediction (x0) at every model call, then decodes them and reports how alpha evolves
    over the diffusion."""

    def __init__(self, split_cfg_alpha):
        self.split_cfg_alpha = split_cfg_alpha
        self.trace = []

    def patch(self, model):
        def record(args):
            sides = (args["cond_denoised"], args["uncond_denoised"]) if self.split_cfg_alpha else ()
            self.trace.append((args["sigma"].flatten()[0].item(), args["denoised"][:1].detach().clone(),
                               [None if d is None else d[:1].detach().clone() for d in sides]))
            return args["denoised"]

        m = model.clone()
        m.set_model_sampler_post_cfg_function(record)
        return m

    def report(self, model, vae):
        def decode(x0):
            img = vae.decode(model.model.process_latent_out(x0.float()))
            return img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])[:1]

        images = []
        # raw mean / >1 use the unclamped alpha, so CFG overshoot past opaque shows up
        header = "call  sigma      min    mean     <255    <250    <128   raw_mean      >1"
        if self.split_cfg_alpha:
            header += "   cond_mean  cond<255  uncond_mean  uncond<255"
        lines = [header]
        for i, (sigma, x0, sides) in enumerate(self.trace):
            img = decode(x0)
            images.append(img)
            if img.shape[-1] != 4:
                lines.append(f"{i:4d}  {sigma:8.4f}  no alpha channel ({img.shape[-1]} channels)")
                continue
            raw = img[..., 3].float()
            a = (raw.clamp(0, 1) * 255).to(torch.int32)
            line = (f"{i:4d}  {sigma:8.4f}  {a.min().item():4d}  {a.float().mean().item():6.2f}  "
                    f"{(a < 255).float().mean().item() * 100:6.2f}% {(a < 250).float().mean().item() * 100:6.2f}% {(a < 128).float().mean().item() * 100:6.2f}%"
                    f"  {raw.mean().item() * 255:8.2f}  {(raw > 1).float().mean().item() * 100:6.2f}%")
            for side in sides:
                if side is None:
                    line += "         n/a       n/a"
                else:
                    sa = (decode(side)[..., 3].clamp(0, 1) * 255).to(torch.int32)
                    line += f"   {sa.float().mean().item():9.2f}  {(sa < 255).float().mean().item() * 100:7.2f}%"
            lines.append(line)

        images = torch.cat(images)
        if images.shape[-1] == 4:
            alpha_maps = images[..., 3:4].clamp(0, 1).expand(-1, -1, -1, 3)
        else:
            alpha_maps = torch.ones_like(images[..., :3])
        stats = "\n".join(lines)
        print(stats)
        return (images[..., :3], alpha_maps, stats)


TRACE_RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
TRACE_RETURN_NAMES = ("x0_images", "alpha_maps", "alpha_stats")
SPLIT_CFG_INPUT = ("BOOLEAN", {"default": False, "tooltip": "Also decode the positive (cond) and negative (uncond) predictions separately. Two extra VAE decodes per step."})


class KSamplerAlphaTrace:
    """KSampler with the alpha trace attached."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_cfg_alpha": SPLIT_CFG_INPUT,
            }
        }

    RETURN_TYPES = ("LATENT",) + TRACE_RETURN_TYPES
    RETURN_NAMES = ("latent",) + TRACE_RETURN_NAMES
    FUNCTION = "sample"
    CATEGORY = "alpha trace"

    def sample(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, split_cfg_alpha):
        tracer = AlphaTracer(split_cfg_alpha)
        m = tracer.patch(model)
        latent = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        return (latent,) + tracer.report(m, vae)


class SamplerCustomAdvancedAlphaTrace:
    """SamplerCustomAdvanced with the alpha trace attached: any noise, guider, sampler and sigmas."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "split_cfg_alpha": SPLIT_CFG_INPUT,
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT") + TRACE_RETURN_TYPES
    RETURN_NAMES = ("output", "denoised_output") + TRACE_RETURN_NAMES
    FUNCTION = "sample"
    CATEGORY = "alpha trace"

    def sample(self, noise, guider, sampler, sigmas, latent_image, vae, split_cfg_alpha):
        tracer = AlphaTracer(split_cfg_alpha)
        # the guider copies model_options from its patcher when built, so trace through a copy with a patched clone
        traced = copy.copy(guider)
        traced.model_patcher = tracer.patch(guider.model_patcher)
        traced.model_options = traced.model_patcher.model_options
        out, out_denoised = SamplerCustomAdvanced.execute(noise, traced, sampler, sigmas, latent_image).result
        return (out, out_denoised) + tracer.report(traced.model_patcher, vae)


HEAT_STOPS = [(0.0, 0.0, 0.0), (0.35, 0.05, 0.55), (0.85, 0.2, 0.2), (1.0, 0.75, 0.1), (1.0, 1.0, 0.85)]


def heat_colormap(h):
    stops = torch.tensor(HEAT_STOPS, device=h.device, dtype=h.dtype)
    pos = h.clamp(0, 1) * (len(HEAT_STOPS) - 1)
    lo = pos.floor().long().clamp(max=len(HEAT_STOPS) - 2)
    t = (pos - lo).unsqueeze(-1)
    return torch.lerp(stops[lo], stops[lo + 1], t)


class AlphaTraceOverlay:
    """Accumulates how far alpha fell short of opaque across every traced step and overlays it on an image."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_maps": ("IMAGE",),
                "skip_first_steps": ("INT", {"default": 1, "min": 0, "max": 10000, "tooltip": "Step 0 is a prediction from pure noise and dominates the sum."}),
                "normalize_percentile": ("FLOAT", {"default": 99.5, "min": 50.0, "max": 100.0, "step": 0.1, "tooltip": "Heat is scaled so this percentile maps to full intensity."}),
                "overlay_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side_by_side", "overlay", "heatmap")
    FUNCTION = "overlay"
    CATEGORY = "alpha trace"

    def overlay(self, image, alpha_maps, skip_first_steps, normalize_percentile, overlay_strength):
        image = image[:1, ..., :3]
        deficit = (1.0 - alpha_maps[skip_first_steps:, ..., 0]).clamp(min=0).sum(dim=0)
        if deficit.shape != image.shape[1:3]:
            deficit = torch.nn.functional.interpolate(deficit[None, None], size=image.shape[1:3], mode="bilinear")[0, 0]
        scale = torch.quantile(deficit.flatten().float(), normalize_percentile / 100.0).clamp(min=1e-6)
        heat = (deficit / scale).clamp(0, 1).to(image.device)

        heatmap = heat_colormap(heat)[None]
        k = (heat * overlay_strength)[None, ..., None]
        overlay = image * (1 - k) + heatmap * k
        return (torch.cat([image, overlay], dim=2), overlay, heatmap)


NODE_CLASS_MAPPINGS = {
    "AlphaTrace_KSampler": KSamplerAlphaTrace,
    "AlphaTrace_SamplerCustomAdvanced": SamplerCustomAdvancedAlphaTrace,
    "AlphaTrace_Overlay": AlphaTraceOverlay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaTrace_KSampler": "KSampler (Alpha Trace)",
    "AlphaTrace_SamplerCustomAdvanced": "SamplerCustomAdvanced (Alpha Trace)",
    "AlphaTrace_Overlay": "Alpha Trace Overlay",
}
