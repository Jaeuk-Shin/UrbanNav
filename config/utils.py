from omegaconf import OmegaConf


def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def load_and_merge_configs(base_config_path, rl_config_path):
    rl_cfg = OmegaConf.load(rl_config_path)
    enc_cfg = OmegaConf.load(base_config_path)

    # Convert to plain dicts for Ray RLlib compatibility
    rl_cfg_dict = OmegaConf.to_container(rl_cfg, resolve=True)
    enc_cfg_dict = OmegaConf.to_container(enc_cfg, resolve=True)

    if "model_config" not in rl_cfg_dict:
        rl_cfg_dict["model_config"] = {}

    rl_cfg_dict["model_config"]['base_config'] = enc_cfg_dict
    return rl_cfg_dict
