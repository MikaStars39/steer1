def steer(
    steer_method: str,
):
    if steer_method == "caa":
        from .caa import caa
        return caa
    elif steer_method == "ablation":
        from .ablation import directional_ablation
        return directional_ablation