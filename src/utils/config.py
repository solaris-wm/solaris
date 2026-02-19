import importlib


def instantiate_from_config(config, **kwargs):
    params = dict(config.get("params", {}))
    additional_params = params.pop("additional_params", {})
    params.update(additional_params)
    return get_obj_from_str(config["target"])(**params, **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
