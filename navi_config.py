class NaviConfig:
    landmarks = {
        "fridge": (3, 4, 0),
        "kitchen": (0, 4, 0),
        "banana": (2, 4, 0),
        "snack": (0, 0, 180),
        "desk": (-2, -4, 180),
        "tv": (-3, -3, 270),
        "curtain": (-2, 2, 180),
        "sofa": (0, -1, 90),
    }
    border_size = 4
    grid_size = 8
    obstacles = {
        "obstacle2": (-3, 1, 0),
        "obstacle3": (-2, 1, 0),
        "obstacle4": (-1, 1, 0),
        "obstacle5": (1, 0, 0),
        "obstacle6": (2, 0, 0),
        "obstacle7": (3, 0, 0),
        "obstacle8": (1, -1, 0),
        "obstacle9": (2, -1, 0),
        "obstacle10": (3, -1, 0),
        "obstacle11": (1, -2, 0),
        "obstacle12": (2, -2, 0),
        "obstacle13": (3, -2, 0),
        "obstacle14": (1, -3, 0),
        "obstacle15": (2, -3, 0),
        "obstacle16": (3, -3, 0),
        "obstacle17": (1, -4, 0),
        "obstacle18": (2, -4, 0),
        "obstacle19": (3, -4, 0),
        "obstacle20": (0, -4, 0),
        "obstacle21": (-1, -4, 0),
        "obstacle22": (-3, -4, 0), 
    }
    # Manually update 'calculate_distance' in openai_client.py
    banana_bottom_left = (-1, -3)
    banana_width = 2
    banana_height = 5
    fridge_bottom_left = (-1, 2)
    fridge_width = 2
    fridge_height = 2
    snack2_bottom_left = (1, 3)
    snack2_width = 2
    snack2_height = 1
    snack1_bottom_left = (-3, -3)
    snack1_width = 2
    snack1_height = 2
    sofa_bottom_left = (-2, -3)
    sofa_width = 1
    sofa_height = 1
    desk_bottom_left = (-1, -3)
    desk_width = 1
    desk_height = 1
    tv_bottom_left = (-1, -3)
    tv_width = 1
    tv_height = 1