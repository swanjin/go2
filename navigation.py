import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

class NaviModel:
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
        goal_orientation = (target_position[2] + 180) % 360
        goal = (target_position[0], target_position[1], goal_orientation)
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

class PathAnimator:
    def __init__(self, start, goal, path, landmarks, obstacles):
        self.start = start
        self.goal = goal
        self.path = path
        self.landmarks = landmarks
        self.obstacles = obstacles
        self.grid_size = 7
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.robot_marker = None
        self.path_points = [(start[0], start[1])]
        self.path_lines = []

    def setup_plot(self):
        self.ax.set_xlim(-self.grid_size, self.grid_size)
        self.ax.set_ylim(-self.grid_size, self.grid_size)
        self.ax.set_xticks(range(-self.grid_size, self.grid_size + 1))
        self.ax.set_yticks(range(-self.grid_size, self.grid_size + 1))
        self.ax.grid(True, color='lightgray', linestyle='-', alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=0.5)
        self.ax.axvline(x=0, color='black', linewidth=0.5)

        for name, (x, y, _) in self.landmarks.items():
            self.ax.plot(x, y, 'ro')
            self.ax.text(x, y, f' {name}', fontsize=9, verticalalignment='bottom')

        first_obstacle = True
        for _, (x, y, _) in self.obstacles.items():
            if first_obstacle:
                self.ax.plot(x, y, 'x', color='grey', markersize=6, label='obstacle')
                first_obstacle = False
            else:
                self.ax.plot(x, y, 'x', color='grey', markersize=6)

        self.ax.plot(self.start[0], self.start[1], 'go', label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'bo', label='Goal')
        self.ax.legend()

    def update(self, frame):
        current_position = list(self.start)

        if frame > 0:
            for i in range(frame):
                prev_x, prev_y = current_position[0], current_position[1]
                current_position = list(NaviModel.get_next_position(current_position, self.path[i]))

                if "turn" not in self.path[i]:
                    new_x, new_y = int(current_position[0]), int(current_position[1])
                    if prev_x != new_x:
                        self.path_points.append((new_x, int(prev_y)))
                    if prev_y != new_y:
                        self.path_points.append((new_x, new_y))

        if self.robot_marker:
            self.robot_marker.remove()
        for line in self.path_lines:
            line.remove()
        self.path_lines.clear()

        if len(self.path_points) > 1:
            for i in range(len(self.path_points) - 1):
                x1, y1 = self.path_points[i]
                x2, y2 = self.path_points[i + 1]
                if x1 == x2 or y1 == y2:
                    line, = self.ax.plot([x1, x2], [y1, y2], 'k:', alpha=0.5)
                    self.path_lines.append(line)

        triangle_size = 0.3
        if current_position[2] == 0:
            triangle = [(current_position[0], current_position[1] + triangle_size),
                        (current_position[0] - triangle_size, current_position[1] - triangle_size),
                        (current_position[0] + triangle_size, current_position[1] - triangle_size)]
        elif current_position[2] == 90:
            triangle = [(current_position[0] + triangle_size, current_position[1]),
                        (current_position[0] - triangle_size, current_position[1] + triangle_size),
                        (current_position[0] - triangle_size, current_position[1] - triangle_size)]
        elif current_position[2] == 180:
            triangle = [(current_position[0], current_position[1] - triangle_size),
                        (current_position[0] - triangle_size, current_position[1] + triangle_size),
                        (current_position[0] + triangle_size, current_position[1] + triangle_size)]
        else:
            triangle = [(current_position[0] - triangle_size, current_position[1]),
                        (current_position[0] + triangle_size, current_position[1] + triangle_size),
                        (current_position[0] + triangle_size, current_position[1] - triangle_size)]

        self.robot_marker = Polygon(triangle, facecolor='black', edgecolor='black')
        self.ax.add_patch(self.robot_marker)

        return (self.robot_marker, *self.path_lines)

    def animate(self):
        self.setup_plot()
        ani = FuncAnimation(self.fig, self.update, frames=len(self.path) + 1, blit=True, repeat=False, interval=500)
        plt.show()

class Mapping:
    def __init__(self):
        self.landmarks = {
            "refrigerator": (4, 4, 180),
            "kitchen": (0, 4, 180),
            "tv": (-5, -1, 90),
            "desk": (-3, -6, 0),
            "cabinet": (0, -6, 0),
            "sofa": (4, -3, 270),
            "banana": (2, 4, 180),
            "bottle": (4, 0, 270)
        }
        self.obstacles = {"obstacle": (-1, 0, 0)}
        self.excluded_points = {(l[0], l[1]) for l in self.landmarks.values()}
        self.border_points = self.generate_border_points()
        self.add_border_obstacles()

    def generate_border_points(self):
        border_points = []
        for i in range(-5, 5):
            border_points.extend([
                (i, 4),    # top border
                (i, -6),   # bottom border
                (-5, i-1), # left border
                (4, i-1)   # right border
            ])
        return border_points

    def add_border_obstacles(self):
        for idx, point in enumerate(self.border_points):
            if (point[0], point[1]) not in self.excluded_points:
                self.obstacles[f"border_{idx}"] = (point[0], point[1], 0)

# Initialize Mapping
mapping = Mapping()

# Run simulation with obstacle avoidance
robot = NaviModel((0, 0, 180))
target = mapping.landmarks["tv"]
path_to_target = robot.navigate_to(target, mapping.obstacles)

# Run animation
animator = PathAnimator(robot.position, target, path_to_target, mapping.landmarks, mapping.obstacles)
animator.animate()