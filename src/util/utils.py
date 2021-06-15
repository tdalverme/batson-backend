def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list