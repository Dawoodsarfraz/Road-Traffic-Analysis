import cv2


def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f" Mouse clicked at: {x}, {y}")


cv2.namedWindow("Test Window")
cv2.setMouseCallback("Test Window", draw_polygon)

while True:
    img = cv2.imread("./media/images/image.png")  # Load any image
    cv2.imshow("Test Window", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
