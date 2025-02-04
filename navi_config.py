class NaviConfig:
    landmarks = {
        "coffee machine": (-2, 6, 0),
        "refrigerator": (4, 5, 0),
        "sink": (0, 6, 0),
        "tv": (-5, -1, 270),
        "desk": (-5, -6, 180),
        "cabinet": (0, -6, 180),
        "sofa": (4, -3, 90),
        "banana": (1, 6, 0),
        "bottle": (3, 1, 90),
    }
    border_size = 7
    grid_size = 10
    obstacles = {
        "obstacle1": (-7, 2, 0),
        "obstacle2": (-6, 2, 0),
        "obstacle3": (-5, 2, 0),
        "obstacle4": (-4, 2, 0),
        "obstacle5": (-3, 2, 0),
        "obstacle6": (-2, 2, 0),
    }
    banana_bottom_left = (-1, 0)
    banana_width = 3
    banana_height = 4
    refrigerator_bottom_left = (-2, 3)
    refrigerator_width = 4
    refrigerator_height = 4
    bottle_bottom_left = (2, 2)
    bottle_width = 2
    bottle_height = 4
    apple_shift_bottom_left = (2, 1)
    apple_shift_width = 2
    apple_shift_height = 2
    apple_forward_bottom_left = (-2, 3)
    apple_forward_width = 5
    apple_forward_height = 2