try:
    from .nodes import NODE_CLASS_MAPPINGS

    NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except:
    import traceback

    print("\033[1;31m%s\033[0m" % traceback.format_exc())
