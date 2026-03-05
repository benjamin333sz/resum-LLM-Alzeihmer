from langfuse import get_client

_langfuse_client = None


def get_langfuse():
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = get_client()
    return _langfuse_client
