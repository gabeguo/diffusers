import torch
from utils.two_timestep_inference import FluxPipelineTwoTimestep as FluxPipeline


def lsd_loss(transformer, x_t0, t0, t1, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids, dt=5e-3):
    # TODO: get rid of extra args
    # Predict the noise residual
    def v_func(the_x_t0, the_t0, the_t1):
        if the_t1 is not None:
            timestep = torch.stack([the_t0, the_t1], dim=-1)
        else:
            timestep = the_t0
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
        return v_t0_t1
    # TODO: if this doesn't work, have debug logs to print the times
    t0 = t0 / 1000
    t1 = t1 / 1000
    v_t0_t1 = v_func(x_t0, t0, t1)
    time_diff = (t1 - t0).reshape(-1, 1, 1)
    assert len(time_diff.shape) == len(v_t0_t1.shape)
    x_t1 = x_t0 + time_diff * v_t0_t1

    transformer.module.disable_adapters()
    with torch.no_grad():
        # recover the original model
        v_t1_t1 = v_func(x_t1, t1, None).detach()
    transformer.module.enable_adapters()
    
    # TODO: double-check time shifting effect on these calculations, but I think it's fine
    # TODO: double-check about scaling on dt, but I think this is the right way
    # TODO: can try central difference for more accuracy
    d_t1_v_t0_t1 = (
        v_func(x_t0, t0, t1 + dt).detach() - v_func(x_t0, t0, t1 - dt).detach()
    ) / (2 * dt)
    d_t1_v_t0_t1 = d_t1_v_t0_t1.detach() # TODO: can remove if we have more memory

    diff = v_t0_t1 + time_diff * d_t1_v_t0_t1 - v_t1_t1.detach()
    # TODO: weighting?
    loss = torch.abs(diff).mean()

    return loss