import math

def output_size_for_input(in_size, depth, padding = True):
    if padding == True:
        for _ in range(depth - 1):
            in_size = in_size // 2
        for _ in range(depth - 1):
            in_size = in_size * 2
        return in_size
    else:
        in_size -= 4
        for _ in range(depth - 1):
            in_size = math.ceil((in_size - 2) / 2) + 1
            in_size -= 4
        for _ in range(depth - 1):
            in_size = in_size * 2
            in_size -= 4
        return int(in_size)
