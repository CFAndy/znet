
def output_size_for_input(in_size, depth):
    for _ in range(depth - 1):
        in_size = in_size // 2
    for _ in range(depth - 1):
        in_size = in_size * 2
    return in_size


