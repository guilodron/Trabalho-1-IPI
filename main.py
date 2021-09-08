import cv2
import numpy as np

# \brief Checa se a imagem está em escala de cinza
# 
# \param original_image Imagem a ser checada
# \return valor booleano representando a checagem
def isGray(original_image):
    return True if len(original_image.shape) < 3 else False

# \brief Reduz a imagem por um fator de 2
# 
# \param original_image Imagem a ser reduzida
# \return Matriz representando a imagem reduzida
def reduceImage(original_image):
    dimensions = 1 if isGray(original_image) else 3
    reduced = np.zeros((int(original_image.shape[0]/2),int(original_image.shape[1]/2), dimensions), dtype=np.uint8)
    x, y = 0, 0
    for i in range(0, original_image.shape[0], 2):
        y = 0
        for j in range(0, original_image.shape[1], 2):
            reduced[x][y] = original_image[i][j]

            y += 1 
        x += 1
    return reduced

# \brief Interpola a imagem por um fator de 2
# 
# \param original_image Imagem a ser interpolada
# \return Matriz representando a imagem interpolada
def interpolate(original_image):
    dimensions = 1 if isGray(original_image) else 3
    interpolated = np.zeros((int(original_image.shape[0]*2),int(original_image.shape[1]*2), dimensions), dtype=np.uint8)
    x, y = 0, 0
    for i in range(0, original_image.shape[0], 1):
        y = 0
        for j in range(0, original_image.shape[1] ,1):
            interpolated[x][y] = original_image[i][j]
            interpolated[x][y+1] = original_image[i][j]
            interpolated[x+1][y] = original_image[i][j]
            interpolated[x+1][y+1] = original_image[i][j]
            y += 2
        x += 2
    return interpolated

# \brief Executa operações de redução e interpolação
# 
# \param original_image Imagem a ser transformada
# \return Matriz representando a imagem que sofreu as duas operações
def dec_int(original_image):
    reduzida = reduceImage(original_image)
    interpolada = interpolate(reduzida)
    return interpolada

original = cv2.imread('images/test80.jpg')
compressed = cv2.resize(original, (int(original.shape[1]/2), int(original.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
interpolated = cv2.resize(compressed, (int(compressed.shape[1]*2), int(compressed.shape[0]*2)), interpolation = cv2.INTER_CUBIC)

my_interpolation = dec_int(original)
cv2.imshow('Original', original)
cv2.imshow('Resized opencv', interpolated)
cv2.imshow('My resized', my_interpolation)

cv2.waitKey()
