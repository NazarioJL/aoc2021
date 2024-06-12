ENHANCEMENT_COUNT = 50
BORDER_SIZE = ENHANCEMENT_COUNT
NEIGHBOURS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

with open("input.txt") as f:
    data = f.read()
data = data.split("\n\n")
filter = data[0]
image = data[1].splitlines()
width, height = len(image[0]), len(image)

padding = ENHANCEMENT_COUNT + BORDER_SIZE
for e in range(padding):
    top_border = "".join(["." for v in range(width)])
    image = [top_border] + image + [top_border]
    for r in range(len(image)):
        image[r] = "." + image[r] + "."
    width, height = width + 2, height + 2

for e in range(1, ENHANCEMENT_COUNT + 1):
    new_image = image.copy()
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            subimage = [
                ("1" if image[row + y][col + x] == "#" else "0") for x, y in NEIGHBOURS
            ]
            number = int("".join(subimage), 2)
            pixel = filter[number]
            new_image[row] = new_image[row][:col] + pixel + new_image[row][col + 1 :]
    image = new_image

count = 0
for row in range(BORDER_SIZE, height - BORDER_SIZE):
    for col in range(BORDER_SIZE, width - BORDER_SIZE):
        count = count + (1 if image[row][col] == "#" else 0)

print(count)
