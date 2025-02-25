class NaviConfig:
    landmarks = {
        "coffee machine": (-2, 5, 0), # Kitchen
        "refrigerator": (3, 5, 0), # Kitchen
        "banana": (1, 5, 0), # Kitchen
        "sofa": (4, -3, 90), 
        "bottle": (3, 1, 90), 
    }
    border_size = 7
    grid_size = 10
    obstacles = {
        "obstacle1": (-7, 1, 0),
        "obstacle2": (-6, 1, 0),
        "obstacle3": (-5, 1, 0),
        "obstacle4": (-4, 1, 0),
        "obstacle5": (-3, 1, 0),
        "obstacle6": (-2, 1, 0),
        "obstacle7": (-1, 1, 0),
    }
    # Manually update 'calculate_distance' in openai_client.py
    banana_bottom_left = (-1, -2)
    banana_width = 2
    banana_height = 6
    refrigerator_bottom_left = (-2, 3)
    refrigerator_width = 3
    refrigerator_height = 4
    bottle_bottom_left = (1, 3)
    bottle_width = 3
    bottle_height = 3
    apple_shift_bottom_left = (1, 1)
    apple_shift_width = 3
    apple_shift_height = 1
    apple_forward_bottom_left = (2, 3)
    apple_forward_width = 2
    apple_forward_height = 2
    sofa_bottom_left = (-3, -2)
    sofa_width = 2
    sofa_height = 2
    kitchen_bottom_left = (-1, -1)
    kitchen_width = 2
    kitchen_height = 2
