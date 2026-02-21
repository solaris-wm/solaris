import os
import pprint

import hydra
import wandb
from absl import logging
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict

from src.utils.config import instantiate_from_config
from src.utils.jax import init_jax_distributed


def _resolve_device_paths(cfg):
    """Resolve relative device paths to absolute using Hydra's original cwd."""
    orig_cwd = get_original_cwd()
    path_keys = [
        "data_dir", "eval_data_dir", "pretrained_model_dir",
        "output_dir", "checkpoint_dir", "jax_cache_dir",
    ]
    with open_dict(cfg):
        for key in path_keys:
            if key in cfg.device:
                val = cfg.device[key]
                if val and not os.path.isabs(val):
                    cfg.device[key] = os.path.join(orig_cwd, val)


@hydra.main(
    config_path="../config", config_name="inference", version_base=None
)  # no version to avoid warnings in cli
def main(cfg):
    # Hydra changes cwd, so resolve relative paths before anything else.
    _resolve_device_paths(cfg)

    # Enable JAX compilation cache
    # NOTE: we should import jax cache AFTER setting up the environment variables.
    import jax

    from src.utils.jax import setup_jax_cache

    if cfg.enable_jax_cache:
        setup_jax_cache(cfg.device.jax_cache_dir)

    logging.info("Setting up jax distributed")
    init_jax_distributed(cfg)

    if not (jax.process_index() == 0):  # not first process
        logging.set_verbosity(logging.ERROR)  # disable info/warning

    inference = instantiate_from_config(cfg.runner)
    inference.run()


if __name__ == "__main__":
    main()
