from HopliteSolver import ioManager

class Solver():

    def posizioniIntorno(self,pos):
        return [(pos[0]+dx, pos[1]+dy) for dx, dy in [(0,1),(1,0),(1,-1),(0,-1),(-1,0),(-1,1)]]

    def posizioniAssi(self,pos, length=5):
        x, y = pos
        res = [(x+i, y) for i in range(-length, length+1, 1)]
        res += [(x, y+i) for i in range(-length, length+1, 1)]
        res += [(x+i, y-i) for i in range(-length, length+1, 1)]
        return res

    def __init__(self):
        self.image = ioManager.getScreenshot()
        self.gray_image = ioManager.convertImage(self.image)
        self.entities_positions = ioManager.detectEntities(self.gray_image)
        lavapos = ioManager.getLava(self.image)
        for pos in lavapos:
            if pos in self.entities_positions:
                raise RuntimeWarning("Posizione già occupata")
            self.entities_positions[pos] = 'lava'
        print(self.entities_positions)
        print(len(self.entities_positions))

#cv2.circle(imageBGR, center, 70, values[0], 2)
# #setX(cartesian)
# if 'mago' in nomefile:
#     for p in posizioniIntorno(cartesian):
#         setX(p)
#     for p in posizioniAssi(cartesian):
#         setX(p)
# # elif 'guerriero' in nomefile or 'bomba_' in nomefile:
# #     for p in posizioniIntorno(cartesian):
# #         setX(p)
# elif 'arcere' in nomefile:
#     for p in posizioniAssi(cartesian):
#         if p not in posizioniIntorno(cartesian):
#             setX(p)