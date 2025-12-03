import torch
from utils.two_timestep_inference import FluxPipelineTwoTimestep as FluxPipeline


def lsd_loss(transformer, x_t0, t0, t1, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids, vae_scale_factor, height, width):
    # Predict the noise residual
    def v_func(the_x_t0, the_t0, the_t1):
        if the_t1 is not None:
            timestep = torch.stack([the_t0 / 1000, the_t1 / 1000], dim=-1)
        else:
            timestep = the_t0 / 1000
        v_t0_t1 = transformer(
            hidden_states=the_x_t0,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        v_t0_t1 = FluxPipeline._unpack_latents(
            v_t0_t1,
            height=height * vae_scale_factor,
            width=width * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        return v_t0_t1
    v_t0_t1, d_t1_v_t0_t1 = torch.func.jvp(
        v_func, 
        primals=(x_t0, t0, t1,),
        tangents=(torch.zeros_like(x_t0), torch.zeros_like(t0), torch.ones_like(t1),),
    )
    x_t1 = x_t0 + (t1 - t0) * v_t0_t1

    with transformer.disable_adapters(), torch.no_grad():
        # recover the original model
        v_t1_t1 = v_func(x_t1, t1, None).detach()
    transformer.enable_adapters()

    diff = v_t0_t1 + (t1 - t0) * d_t1_v_t0_t1 - v_t1_t1.detach()
    # TODO: weighting?
    loss = torch.abs(diff).mean()

    return loss