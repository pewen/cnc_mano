import math

import numpy as np
from cv2 import getRotationMatrix2D, warpAffine


def decimal2encoding(number):
    """
    Crea la imagen para un numero dado

    Parameter
    ---------
    number: int
      Deceminal a codificar

    Return
    ------
    img: numpy.matrix
      Matriz de la codificacion
    """
    # Resolucion de la imagen
    img = np.zeros((100, 220))

    # Se hace el tringulo en la esquina superior izquierda
    for row in range(20):
        for col in range(20):
            if row + col < 20:
                img[row, col] = 255

    binary = bin(number)[2:]
    # Se completa con ceros para que tenga el mismo largo siempre
    if len(binary) < 10:
        binary = '0' * (10 - len(binary)) + binary

    x_pos = 0
    y_pos = 20

    for i, bit in enumerate(str(binary)):
        x_pos += 20
        if i == 5:
            y_pos = 60
            x_pos = 20
        if bit == '1':
            img[y_pos:y_pos+20, x_pos:x_pos+20] = 255
        x_pos += 20

    return img


def makeTape(start, quantity):
    """
    Genera una cinta con n (quantity) de elemento ordenados
    comenzando desde start

    Parameters
    ----------
    start: int
        Punto de comienzo de la cinta
    quantity: int
        Cantidad de elementos de la cinta

    Return
    ------
    cinta: numpy.matrix
    """
    largo = (40 + 220) * quantity
    ancho = 100 + 40
    cinta = np.ones((ancho, largo)) * 255
    x_pos = 20
    y_pos = 20

    for i in range(quantity):
        cinta[y_pos:y_pos+100, x_pos:x_pos+220] = decimal2encoding(start + i)
        x_pos += 40 + 220

    return cinta


def boxAngle(box):
    """
    Angulo de un rectangulo con respecto al eje mas grande de este.

    Parameter
    ---------
    box: cv2.boxPoints()
        Punto del rectangulo

    Return
    ------
    angle: float
        Angulo del rectangulo
    """
    # Calulo los dos ejes
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]

    # Me quedo con el eje mayor
    edge1_norm = np.linalg.norm(edge1)
    edge2_norm = np.linalg.norm(edge2)
    usedEdge = edge1
    if edge2_norm > edge1_norm:
        usedEdge = edge2

    # Horizontal edge
    reference = np.array([1, 0])
    # norms
    ref_norm = np.linalg.norm(reference)
    usedEdge_norm = np.linalg.norm(usedEdge)
    product = reference[0]*usedEdge[0] + reference[1]*usedEdge[1]

    angle = 180/math.pi * math.acos(product / (ref_norm*usedEdge_norm))

    return angle


def rotateAndScale(img, angle, scaleFactor=1):
    """
    Roatate and scale a image

    Parameters
    img: numpy.matrix
        Image
    angle: float
        Angle in degree to roatate the image
    scaleFactor: float, optional
        Scale factor
    """
    # Note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    oldY, oldX = img.shape
    # Rotate about center of image.
    rot_matrix = getRotationMatrix2D(center=(oldX/2, oldY/2),
                                     angle=angle, scale=scaleFactor)

    # Choose a new image size.
    newX, newY = oldX*scaleFactor, oldY*scaleFactor
    # Include this if you want to prevent corners being cut off
    r = np.deg2rad(angle)
    newX, newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),
                  abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    # The warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion
    # of the resulting image.

    # So I will find the translation that moves the result to the
    # center of that region.
    tx, ty = ((newX-oldX)/2, (newY-oldY)/2)

    # Third column of matrix holds translation,
    # which takes effect after rotation.
    rot_matrix[0, 2] += tx
    rot_matrix[1, 2] += ty

    rotatedImg = warpAffine(img, rot_matrix, dsize=(int(newX), int(newY)))

    return rotatedImg


def encoding2binary(image, box):
    """
    Representacion binaria de la imagen codificada

    Parameters
    ----------
    image:

    box: cv2.boxPoints()
        Punto del rectangulo

    Return
    ------
    binary: str
        representacion binaria
    """
    # Si la imagen esta rotada, la pongo horizontal
    angle = boxAngle(box)
    if angle != 0:
        image = rotateAndScale(image, angle)

    # Delta y offset dependen de la resolucion de la imagen
    delta1 = max(image.shape)/11
    delta2 = min(image.shape)/5
    delta = int((delta1 + delta2)/2)
    offset = delta + int(delta/2)

    # Check the orientation of the image
    index = int(delta/4)
    orientation = image[index, index]
    if not orientation:
        rows, cols = image.shape

        rot_matrix = getRotationMatrix2D((cols/2, rows/2), 180, 1)
        image = warpAffine(image, rot_matrix, (cols, rows))

    x = [offset + 2*delta*i for i in range(5)]
    y = [offset + 2*delta*i for i in range(2)]

    binary = ''
    for pixel_y in y:
        for pixel_x in x:
            value = image[pixel_y, pixel_x]
            binary += str(int(value/255))

    return binary
