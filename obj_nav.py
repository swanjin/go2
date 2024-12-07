import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

class Go2Robot:
    def __init__(self, start_position):
        self.position = start_position  # (x, y, orientation)

    @staticmethod
    def heuristic(a, b):
        """Manhattan distance heuristic for grid navigation"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def get_next_position(current_position, action):
        """Calculate next position based on current position and action"""
        x, y, orientation = current_position
        next_position = list(current_position)

        if "move forward" in action:
            if orientation == 0:
                next_position[1] += 1
            elif orientation == 90:
                next_position[0] += 1
            elif orientation == 180:
                next_position[1] -= 1
            elif orientation == 270:
                next_position[0] -= 1
        elif "move backward" in action:
            if orientation == 0:
                next_position[1] -= 1
            elif orientation == 90:
                next_position[0] -= 1
            elif orientation == 180:
                next_position[1] += 1
            elif orientation == 270:
                next_position[0] += 1
        elif "shift right" in action:
            if orientation == 0:
                next_position[0] += 1
            elif orientation == 90:
                next_position[1] -= 1
            elif orientation == 180:
                next_position[0] -= 1
            elif orientation == 270:
                next_position[1] += 1
        elif "shift left" in action:
            if orientation == 0:
                next_position[0] -= 1
            elif orientation == 90:
                next_position[1] += 1
            elif orientation == 180:
                next_position[0] += 1
            elif orientation == 270:
                next_position[1] -= 1
        elif "turn right" in action:
            next_position[2] = (orientation + 90) % 360
        elif "turn left" in action:
            next_position[2] = (orientation - 90) % 360

        return tuple(next_position)

    def is_valid_position(self, position, obstacles):
        """Check if position is valid (not colliding with obstacles)"""
        x, y = position[0], position[1]
        for obs_pos in obstacles.values():
            if (x, y) == (obs_pos[0], obs_pos[1]):
                return False
        return True

    def get_neighbors(self, current, obstacles):
        """Generate possible moves based on the robot's orientation"""
        actions = ["move forward", "move backward", "shift right", "shift left", "turn right", "turn left"]
        neighbors = []
        for action in actions:
            next_pos = self.get_next_position(current, action)
            if "turn" in action or self.is_valid_position(next_pos, obstacles):
                neighbors.append((next_pos, action))
        return neighbors

    def navigate_to(self, target_position, obstacles):
        """A* pathfinding to target position with obstacle avoidance"""
        start = self.position
        goal = target_position
        open_set = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        actions = {}

        while open_set:
            _, current = heapq.heappop(open_set)

            if (current[0], current[1]) == (goal[0], goal[1]):
                if current[2] == goal[2]:
                    break

            for neighbor, action in self.get_neighbors(current, obstacles):
                action_cost = 1 if "turn" in action else 2 if "move" in action else 3
                if "move" in action and current[2] != goal[2]:
                    continue

                new_cost = cost_so_far[current] + action_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic((neighbor[0], neighbor[1]), (goal[0], goal[1]))
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
                    actions[neighbor] = action

        path = []
        current = goal
        while current != start:
            if current not in came_from:
                break
            path.append(actions[current])
            current = came_from[current]
        
        return list(reversed(path))

def animate_path(start, goal, path, landmarks, obstacles):
    """Animate the robot's path and save it to a file"""
    # Setup plot
    grid_size = 7
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)
    ax.set_xticks(range(-grid_size, grid_size + 1))
    ax.set_yticks(range(-grid_size, grid_size + 1))
    
    # Set grid style
    ax.grid(True, color='lightgray', linestyle='-', alpha=0.3)  # Light gray grid
    
    # Add black axes at x=0 and y=0
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Plot landmarks, start/goal, and obstacles
    for name, (x, y, _) in landmarks.items():
        ax.plot(x, y, 'ro')
        ax.text(x, y, f' {name}', fontsize=9, verticalalignment='bottom')
    
    # Plot obstacles in grey with smaller 'x' marker
    first_obstacle = True
    for _, (x, y, _) in obstacles.items():
        if first_obstacle:
            ax.plot(x, y, 'x', color='grey', markersize=6, label='obstacle')
            first_obstacle = False
        else:
            ax.plot(x, y, 'x', color='grey', markersize=6)
    
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'bo', label='Goal')

    robot_marker = None
    path_points = [(start[0], start[1])]  # Initialize with starting point
    path_lines = []

    def update(frame):
        nonlocal robot_marker, path_lines
        current_position = list(start)
        
        if frame > 0:
            for i in range(frame):
                prev_x, prev_y = current_position[0], current_position[1]
                current_position = list(Go2Robot.get_next_position(current_position, path[i]))
                
                if "turn" not in path[i]:
                    new_x, new_y = int(current_position[0]), int(current_position[1])
                    if prev_x != new_x:
                        path_points.append((new_x, int(prev_y)))
                    if prev_y != new_y:
                        path_points.append((new_x, new_y))

        # Update visualization
        if robot_marker:
            robot_marker.remove()
        for line in path_lines:
            line.remove()
        path_lines.clear()

        # Draw path
        if len(path_points) > 1:
            for i in range(len(path_points) - 1):
                x1, y1 = path_points[i]
                x2, y2 = path_points[i + 1]
                if x1 == x2 or y1 == y2:
                    line, = ax.plot([x1, x2], [y1, y2], 'k:', alpha=0.5)
                    path_lines.append(line)

        # Draw robot triangle
        triangle_size = 0.3
        if current_position[2] == 0:  # Up
            triangle = [(current_position[0], current_position[1] + triangle_size),
                       (current_position[0] - triangle_size, current_position[1] - triangle_size),
                       (current_position[0] + triangle_size, current_position[1] - triangle_size)]
        elif current_position[2] == 90:  # Right
            triangle = [(current_position[0] + triangle_size, current_position[1]),
                       (current_position[0] - triangle_size, current_position[1] + triangle_size),
                       (current_position[0] - triangle_size, current_position[1] - triangle_size)]
        elif current_position[2] == 180:  # Down
            triangle = [(current_position[0], current_position[1] - triangle_size),
                       (current_position[0] - triangle_size, current_position[1] + triangle_size),
                       (current_position[0] + triangle_size, current_position[1] + triangle_size)]
        else:  # Left
            triangle = [(current_position[0] - triangle_size, current_position[1]),
                       (current_position[0] + triangle_size, current_position[1] + triangle_size),
                       (current_position[0] + triangle_size, current_position[1] - triangle_size)]

        robot_marker = Polygon(triangle, facecolor='black', edgecolor='black')
        ax.add_patch(robot_marker)

        return (robot_marker, *path_lines)

    ani = FuncAnimation(fig, update, frames=len(path) + 1, blit=True, repeat=False, interval=500)
    ax.legend()
    
    # Save the animation
    ani.save('robot_path_animation.mp4', writer='ffmpeg', fps=2)
    
    plt.show()

# Define landmarks and obstacles
landmarks = {
    "refrigerator": (4, 4, 0),
    "kitchen": (0, 4, 0),
    "tv": (-5, 0, 270),
    "desk": (-3, -6, 180),
    "cabinet": (0, -6, 180),
    "sofa": (4, -3, 90),
    "apple": (-5, 4, 270),
    "banana": (2, 4, 0),
    "bottle": (4, 0, 0)
}

obstacles = {
    "obstacle": (-1, 0, 0)
}

# Add border no-go zones
excluded_points = {(l[0], l[1]) for l in landmarks.values()}
border_points = []

# Generate border points
for i in range(-5, 5):
    border_points.extend([
        (i, 4),    # top border
        (i, -6),   # bottom border
        (-5, i-1), # left border
        (4, i-1)   # right border
    ])

# Add border points as obstacles, excluding landmark positions
for idx, point in enumerate(border_points):
    if (point[0], point[1]) not in excluded_points:
        obstacles[f"border_{idx}"] = (point[0], point[1], 0)

# Run simulation with obstacle avoidance
robot = Go2Robot((0, 0, 180))
target = landmarks["tv"]
path_to_target = robot.navigate_to(target, obstacles)

print(path_to_target)  # Print the list of actions directly

# Run animation
animate_path(robot.position, target, path_to_target, landmarks, obstacles)
