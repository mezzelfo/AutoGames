from networkx.algorithms import bipartite
import numpy as np
import cv2
from ppadb.client import Client as AdbClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

img = get_screencap(device)

borders = cv2.Laplacian(img, cv2.CV_64F)
borders = np.any(borders, -1)
borders = 255*borders.astype(np.uint8)
borders[:120, :] = 0

kernel = np.ones((15, 15))

fixed = cv2.morphologyEx(borders, cv2.MORPH_BLACKHAT, kernel)
grid = cv2.subtract(borders, cv2.dilate(fixed, kernel))
blocks = cv2.subtract(255, cv2.dilate(grid, kernel))

contours, _ = cv2.findContours(
    blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[1:-1]  # remove first and last (bars below and above grid)

# cv2.imshow('img',img)
# cv2.imshow('borders',borders)
# cv2.imshow('fixed',fixed)
# cv2.imshow('grid',grid)
# cv2.imshow('blocks',blocks)
# cv2.waitKey()
# exit()

extracted_data = []
for c in contours:
    mask = np.zeros_like(borders)
    cv2.drawContours(mask, [c], -1, 255, -1)
    isfixed = cv2.mean(fixed, mask=mask)[0] > 0
    color = cv2.mean(img, mask=cv2.subtract(mask, fixed))[:-1]
    color = tuple(int(k) for k in color)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    coords = (cX, cY)
    extracted_data.append(
        {
            'coords': coords,
            'color': color,
            'isfixed': isfixed,
            'contour': c
        })

regressor = make_pipeline(PolynomialFeatures(2), LinearRegression())
regressor.fit(
    np.asarray([c['coords'] for c in extracted_data if c['isfixed']]),
    np.asarray([c['color'] for c in extracted_data if c['isfixed']]))

tofill_space_coords = np.asarray([c['coords']
                                 for c in extracted_data if not c['isfixed']])
predicted_colors = regressor.predict(tofill_space_coords).astype(np.int32)
detected_colors = np.asarray(
    [c['color'] for c in extracted_data if not c['isfixed']]).astype(np.int32)
distances = np.linalg.norm(
    (predicted_colors[:, None, :]-detected_colors[None, :, :]), axis=-1)

predicted_colors = [tuple(t) for t in predicted_colors]
detected_colors = [tuple(t) for t in detected_colors]

# blank = np.zeros_like(img)
# for c in extracted_data:
#     cv2.drawContours(blank, [c['contour']], -1, c['color'], -1)
#     if c['isfixed']:
#         cv2.circle(blank, c['coords'],15,(0,0,0),-1)
# cv2.imshow('a',blank)
# cv2.waitKey()
# exit()


G = nx.Graph()
G.add_nodes_from([tuple(['P']+list(c)) for c in predicted_colors], bipartite=0)
G.add_nodes_from(detected_colors, bipartite=1)
for i in range(len(predicted_colors)):
    for j in range(len(detected_colors)):
        G.add_edge(tuple(['P']+list(predicted_colors[i])),
                   detected_colors[j], weight=distances[i, j])

matching = minimum_weight_full_matching(G)
final_position = dict()
actual_position = dict()

for i, color in enumerate(detected_colors):
    target_col = matching[color][1:]
    target_pos = tofill_space_coords[predicted_colors.index(target_col)]

    final_position[color] = tuple(target_pos.tolist())
    actual_position[color] = tuple(tofill_space_coords[i].tolist())

assert len(list(final_position.values())) == len(set(final_position.values()))

while True:
    # search a color not in the right place
    selected = None
    for color in detected_colors:
        if final_position[color] != actual_position[color]:
            selected = color
            break
    if selected is None:
        break

    device.input_swipe(
        *actual_position[selected],
        *final_position[selected],
        1000
    )

    replaced = [c for c in detected_colors if actual_position[c]
                == final_position[selected]]
    assert len(replaced) == 1
    replaced = replaced[0]
    actual_position[replaced] = actual_position[selected]
    actual_position[selected] = final_position[selected]
