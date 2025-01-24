try:
    NODE_CLASS_MAPPINGS = {}
    from .nodes import NODE_CLASS_MAPPINGS as default_NODE_CLASS_MAPPINGS

    NODE_CLASS_MAPPINGS.update(default_NODE_CLASS_MAPPINGS)
    from .sana.nodes import NODE_CLASS_MAPPINGS as sana_NODE_CLASS_MAPPINGS

    NODE_CLASS_MAPPINGS.update(sana_NODE_CLASS_MAPPINGS)

    NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except:
    import traceback

    print("\033[1;31m%s\033[0m" % traceback.format_exc())
