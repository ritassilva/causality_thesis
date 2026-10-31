import os


def avici_experimental_chunk_size():
    return int(os.environ.get("AVICI_EXPERIMENTAL_CHUNK_SIZE", "500"))
