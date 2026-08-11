def adopt_convolution_parameters(target, original):
    """Make a replacement convolution reuse the original Parameters."""
    target.weight = original.weight
    target.bias = original.bias
    return target


def replace_child_convolution(
    module,
    adapter,
    *,
    child="conv",
    conv_block_size=0,
    parallel_context=None,
):
    """Replace a child convolution while giving its weights to the adapter."""
    convolution = getattr(module, child)
    adapted = adapter(
        convolution,
        block_size=conv_block_size,
        parallel_context=parallel_context,
    )
    setattr(module, child, adapted)
    return adapted
