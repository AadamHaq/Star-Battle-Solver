"""
Will research implementing computer vision and add it here
"""

import cv2 as cv



# Load the image
image = cv.imread("Queens/example_board.png")

# Resize to standardise (may remove)
resized = cv.resize(image, (350, 350))

# Convert to grayscale
gray = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
# Create threshold to detect cell boundaries
_, thresh = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

# Find contours which represent the blocks
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a mask for each unique color
unique_colors = {}
board_size = 7  # 7x7 grid in example, need to find a generalisation
cell_height = resized.shape[0] // board_size
cell_width = resized.shape[1] // board_size
board = [[-1 for i in range(board_size)] for j in range(board_size)] # Fi.l board with -1

color_index = 0

# Loop through each cell
for row in range(board_size):
    for col in range(board_size):
        # Get the center pixel of the cell
        y = row * cell_height + cell_height // 2 # Bottom of cell + 1/2 the height to get middle
        x = col * cell_width + cell_width // 2
        color = tuple(resized[y, x])

        # Register new color if not already
        if color not in unique_colors:
            unique_colors[color] = color_index
            color_index += 1

        board[row][col] = unique_colors[color]

print(board)

"""
Output:
[[0, 0, 0, 1, 1, 1, 1],
[2, 0, 3, 3, 3, 1, 1],
[2, 0, 0, 0, 3, 1, 4],
[2, 2, 3, 3, 3, 1, 4],
[5, 5, 3, 6, 6, 4, 4],
[5, 4, 3, 3, 3, 4, 4],
[4, 4, 4, 4, 4, 4, 4]]
"""