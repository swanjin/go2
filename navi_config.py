class NaviConfig:
    landmarks = {
        "banana": (2, 4, 0),
        "refrigerator": (3, 4, 0),
        "kitchen": (0, 4, 0),
        "curtain": (-1, 3, 180),
    }
    border_size = 4
    grid_size = 7
    obstacles = {
        "obstacle1": (-4, 1, 0),
        "obstacle2": (-3, 1, 0),
        "obstacle3": (-2, 1, 0),
        "obstacle4": (-1, 1, 0),
    }
    # Manually update 'calculate_distance' in openai_client.py
    banana_bottom_left = (-1, -3)
    banana_width = 2
    banana_height = 5
    refrigerator_bottom_left = (-1, 2)
    refrigerator_width = 2
    refrigerator_height = 2
    bottle_bottom_left = (1, 3)
    bottle_width = 3
    bottle_height = 1
    sofa_bottom_left = (-3, -3)
    sofa_width = 2
    sofa_height = 2
    # apple_shift_bottom_left = (1, 1)
    # apple_shift_width = 3
    # apple_shift_height = 1
    # apple_forward_bottom_left = (2, 3)
    # apple_forward_width = 2
    # apple_forward_height = 2