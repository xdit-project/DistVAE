def replace_child_convolution(
    module,
    adapter,
    *,
    child="conv",
    conv_block_size=0,
    patch_dim=-2,
    parallel_context=None,
):
    """Replace a child convolution while giving its weights to the adapter."""
    convolution = getattr(module, child)
    adapted = adapter(
        convolution,
        block_size=conv_block_size,
        patch_dim=patch_dim,
        parallel_context=parallel_context,
    )
    setattr(module, child, adapted)
    return adapted
