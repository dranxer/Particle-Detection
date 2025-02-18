import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r"C:\Users\ytewa\Downloads\1739726513410.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
for cnt in contours:
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) == 4:
        cv2.drawContours(result, [approx], -1, (0, 0, 255), 2)
    else:
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

out_path = r"C:\Users\ytewa\Downloads\contours_output.jpg"
cv2.imwrite(out_path, result)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Contours of Major Faces (Rectangles in Red)")
plt.axis("off")
plt.show()

print(f"Saved contour image to: {out_path}")