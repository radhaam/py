import cv2
import pandas as pd

# Load color dataset (you can use your own or this one)
# Download 'colors.csv' with columns: color, color_name, hex, R, G, B
csv_path = 'colors.csv'
index = ["color", "color_name", "hex", "R", "G", "B"]
df = pd.read_csv(csv_path, names=index, header=None)

# Function to get closest color name
def get_color_name(R, G, B):
    minimum = float('inf')
    cname = ""
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname

# Mouse callback function
clicked = False
r = g = b = xpos = ypos = 0

def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Load image
img_path = 'image1.jpeg'
img = cv2.imread(img_path)
img = cv2.resize(img, (800, 600))

cv2.namedWindow('Color Detector')
cv2.setMouseCallback('Color Detector', draw_function)

while True:
    cv2.imshow("Color Detector", img)
    if clicked:
        # Draw rectangle and text
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        text = get_color_name(r, g, b) + f' R={r} G={g} B={b}'
        color = (255, 255, 255) if r + g + b < 600 else (0, 0, 0)
        cv2.putText(img, text, (50, 50), 2, 0.8, color, 2, cv2.LINE_AA)
        clicked = False

    if cv2.waitKey(20) & 0xFF == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
