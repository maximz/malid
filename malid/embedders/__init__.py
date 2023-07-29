def get_embedder_by_name(name):

    # TODO: generate this programatically... by storing names in a central registry, or by introspection on class names? But avoid slow runtime imports.
    # For now, we use tests to confirm these are the same names as the name attributes of these embedders.

    if name == "unirep":
        from .unirep import UnirepEmbedder

        return UnirepEmbedder
    elif name == "unirep_fine_tuned":
        from .unirep import UnirepFineTunedEmbedder

        return UnirepFineTunedEmbedder
    else:
        raise ValueError("Unrecognized embedder type")
