# Steamlit visualization app by Andrew Taylor
# atayl136
# 09/22/24


import heapq
import streamlit as st
from a_star_search import (
    full_world,
    small_world,
    COSTS,
    MOVES,
    heuristic as manhattan_heuristic,
    reconstruct_path
)
from copy import deepcopy
import time
from math import sqrt
import heapq

# add the Euclidean distance heuristic
def euclidean_heuristic(node, goal):
    return sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

# Title and description
st.title("A* Algorithm Demo on Emoji Terrain")
st.write("""
This app demonstrates how the A* algorithm finds the best path through a terrain represented by emojis. Each terrain type has an associated cost.
""")

# Initialize start and goal positions
if 'start' not in st.session_state:
    st.session_state['start'] = None
if 'goal' not in st.session_state:
    st.session_state['goal'] = None

# Select world size
world_option = st.selectbox("Select the world size:", ("Small World", "Full World"))

if world_option == "Small World":
    world = small_world
else:
    world = full_world

world_width = len(world[0])
world_height = len(world)

# Adjust terrain costs 
st.subheader("Terrain Costs")
costs = COSTS.copy()
for terrain in costs.keys():
    costs[terrain] = st.number_input(f"Cost for {terrain}", value=costs[terrain], min_value=1)

# Heuristic selection
st.subheader("Select Heuristic")
heuristic_option = st.selectbox("Heuristic Function:", ("Manhattan Distance", "Euclidean Distance"))

if heuristic_option == "Manhattan Distance":
    heuristic = manhattan_heuristic
else:
    heuristic = euclidean_heuristic

# Display the terrain and allow user to click to select positions
st.subheader("Select Start and Goal Positions by Clicking on the Map")

# Function to get a unique key for each button
def get_button_key(x, y):
    return f"cell_{x}_{y}"

# Create a grid of buttons
for y in range(world_height):
    cols = st.columns(world_width)
    for x in range(world_width):
        terrain = world[y][x]
        button_label = terrain
        button_key = get_button_key(x, y)

        # Highlight the start and goal positions
        if st.session_state['start'] == (x, y):
            button_label = '🚩'
        elif st.session_state['goal'] == (x, y):
            button_label = '🎁'  # Goal emoji changed to '🎁'

        # When a button is clicked, set the start or goal position
        if cols[x].button(button_label, key=button_key):
            if st.session_state['start'] is None:
                st.session_state['start'] = (x, y)
            elif st.session_state['goal'] is None:
                st.session_state['goal'] = (x, y)
            else:
                # Reset positions if both are already set
                st.session_state['start'] = (x, y)
                st.session_state['goal'] = None
            st.experimental_rerun()  # Refresh the app to update the map

# Display current start and goal positions
st.write(f"Current Start Position: {st.session_state['start']}")
st.write(f"Current Goal Position: {st.session_state['goal']}")

# Option to reset positions
if st.button("Reset Start and Goal Positions"):
    st.session_state['start'] = None
    st.session_state['goal'] = None
    st.experimental_rerun()

# Run the A* algorithm when the user clicks the button
if st.button("Run A* Search"):
    if st.session_state['start'] is None or st.session_state['goal'] is None:
        st.error("Please select both a start and goal position by clicking on the map.")
    else:
        with st.spinner("Running A* Search..."):
            # Animate the algorithm
            def a_star_search_animated(world, start, goal, costs, moves, heuristic):
                frontier = [(heuristic(start, goal), start)]
                came_from = {}
                g_score = {start: 0}
                f_score = {start: heuristic(start, goal)}
                explored = set()

                # For visualization
                base_world = deepcopy(world)
                path_display = st.empty()

                while frontier:
                    current = heapq.heappop(frontier)[1]

                    if current == goal:
                        path = reconstruct_path(came_from, current)
                        return path

                    explored.add(current)

                    # Reconstruct path to current node
                    path_to_current = reconstruct_path(came_from, current)

                    # Create a fresh copy of the world for visualization
                    world_copy = deepcopy(base_world)

                    # Overlay the path using directional emojis
                    def overlay_path(world_copy, path, start, goal):
                        total_cost = 0
                        if len(path) > 1:
                            first_move = path[1]
                            if first_move[0] > start[0]:
                                world_copy[start[1]][start[0]] = '⏩'
                            elif first_move[1] > start[1]:
                                world_copy[start[1]][start[0]] = '⏬'
                            elif first_move[0] < start[0]:
                                world_copy[start[1]][start[0]] = '⏪'
                            elif first_move[1] < start[1]:
                                world_copy[start[1]][start[0]] = '⏫'
                            else:
                                world_copy[start[1]][start[0]] = '🚩'
                        else:
                            world_copy[start[1]][start[0]] = '🚩'

                        for i in range(1, len(path)):
                            position = path[i]
                            if position == goal:
                                world_copy[position[1]][position[0]] = '🎁'
                            else:
                                prev_position = path[i-1]
                                if position[0] > prev_position[0]:
                                    world_copy[position[1]][position[0]] = '⏩'
                                elif position[1] > prev_position[1]:
                                    world_copy[position[1]][position[0]] = '⏬'
                                elif position[0] < prev_position[0]:
                                    world_copy[position[1]][position[0]] = '⏪'
                                elif position[1] < prev_position[1]:
                                    world_copy[position[1]][position[0]] = '⏫'
                                else:
                                    world_copy[position[1]][position[0]] = '🔵'  # Fallback
                            total_cost += costs[world[position[1]][position[0]]]

                        return world_copy, total_cost

                    # Overlay the path to current node
                    world_with_path, _ = overlay_path(world_copy, path_to_current, start, goal)

                    # Display the world
                    display_world = "\n".join(["".join(line) for line in world_with_path])
                    path_display.text(display_world)
                    time.sleep(0.1)

                    for move in MOVES:
                        neighbor = (current[0] + move[0], current[1] + move[1])

                        # Check if neighbor is within bounds
                        if 0 <= neighbor[0] < world_width and 0 <= neighbor[1] < world_height:
                            terrain_type = world[neighbor[1]][neighbor[0]]
                            if terrain_type not in costs:
                                continue

                            tentative_g_score = g_score[current] + costs[terrain_type]

                            if neighbor in explored and tentative_g_score >= g_score.get(neighbor, float('inf')):
                                continue

                            if tentative_g_score < g_score.get(neighbor, float('inf')):
                                came_from[neighbor] = current
                                g_score[neighbor] = tentative_g_score
                                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                                heapq.heappush(frontier, (f_score[neighbor], neighbor))

                # If the frontier is empty and goal was not reached
                return []

            path = a_star_search_animated(world, st.session_state['start'], st.session_state['goal'], costs, MOVES, heuristic)

        if path:
            st.success(f"Path found! Total steps: {len(path)-1}")

            # Visualize the final path using directional emojis
            st.subheader("Path Visualization")
            total_cost = 0
            world_copy = deepcopy(world)

            def pretty_print_path(world, path, start, goal, costs):
                total_cost = 0
                world_copy = deepcopy(world)

                # Determine the direction of the first move from start
                if len(path) > 1:
                    first_move = path[1]
                    if first_move[0] > start[0]:
                        world_copy[start[1]][start[0]] = '⏩'
                    elif first_move[1] > start[1]:
                        world_copy[start[1]][start[0]] = '⏬'
                    elif first_move[0] < start[0]:
                        world_copy[start[1]][start[0]] = '⏪'
                    elif first_move[1] < start[1]:
                        world_copy[start[1]][start[0]] = '⏫'
                    else:
                        world_copy[start[1]][start[0]] = '🚩'
                else:
                    world_copy[start[1]][start[0]] = '🚩'

                # Replace the rest of the path with direction symbols
                for i in range(1, len(path)):
                    position = path[i]
                    if position == goal:
                        world_copy[position[1]][position[0]] = '🎁'
                    else:
                        prev_position = path[i-1]
                        if position[0] > prev_position[0]:
                            world_copy[position[1]][position[0]] = '⏩'
                        elif position[1] > prev_position[1]:
                            world_copy[position[1]][position[0]] = '⏬'
                        elif position[0] < prev_position[0]:
                            world_copy[position[1]][position[0]] = '⏪'
                        elif position[1] < prev_position[1]:
                            world_copy[position[1]][position[0]] = '⏫'
                        else:
                            world_copy[position[1]][position[0]] = '🔵'  # Fallback
                    total_cost += costs[world[position[1]][position[0]]]

                return world_copy, total_cost

            # Get the world with the path overlaid and total cost
            world_with_path, total_cost = pretty_print_path(world, path, st.session_state['start'], st.session_state['goal'], costs)

            # Display the world with the path
            path_display = "\n".join(["".join(row) for row in world_with_path])
            st.text(path_display)
            st.write(f"**Total Path Cost:** {total_cost}")
        else:
            st.error("No path found between the selected start and goal positions.")
