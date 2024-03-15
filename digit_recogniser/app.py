import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
pygame.init()

BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
MODEL = load_model("bestmodel.h5")
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

FONT = pygame.font.Font(None, 36)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
drawing_surface = pygame.Surface((WINDOWSIZEX, WINDOWSIZEY))  # Separate surface for drawing
WHITE_INT = WHITE
pygame.display.set_caption("Digit Board")

iswriting = False
PREDICT = True
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = pygame.mouse.get_pos()
            pygame.draw.circle(drawing_surface, WHITE, (xcord, ycord), 8, 0)  # Increase circle size
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.surfarray.array3d(drawing_surface))
            img_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png", img_arr)

            if PREDICT:
                # Resize the image to 28x28
                image = cv2.resize(img_arr, (28, 28))

                # Flatten the image to a 1D array of size 784
                image = image.reshape(1, 784)

                # Normalize the image
                image = image / 255.0  # Assuming pixel values are in the range [0, 255]

                label = str(LABELS[np.argmax(MODEL.predict(image))])

                '''# Resize the image
                image = cv2.resize(img_arr, (28, 28)) / 255

                # Flatten the image
                image = image.reshape(1, 28, 28, 1)

                label = str(LABELS[np.argmax(MODEL.predict(image))])

                image = cv2.resize(img_arr, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])'''

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)

    DISPLAYSURF.blit(drawing_surface, (0, 0))  # Blit drawing surface onto the main surface
    pygame.display.update()
