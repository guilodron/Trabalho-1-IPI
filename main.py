import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    # Descomentar duas linhas abaixo para visualizar redimensionamento codificado
    # cv2.imshow('Reduzida', reduzida)
    # cv2.imshow('Interpolada', interpolada)
    cv2.waitKey()
    return interpolada

# \brief Aplica um filtro de aguçamento Laplaciano na imagem
# 
# \param image imagem a ser filtrada
# \return matriz representando a imagem filtrada
def edge_improv(image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    improved = cv2.subtract(image, filtered)
    return improved

# \brief Aplica a transformação de potência em uma imagem
# 
# \param image Imagem a ser transformada
# \param gamma expoente a ser aplicado na transformaçao
# \return Matriz com os valores transformados
def power_law(image, gamma):
    cv2.imshow('Gamma: ' + str(gamma) ,np.array(np.uint8(255*(image/255)**gamma)))

# \brief Realiza a equalizaçao de histograma a fim de balancear a imagem
# 
# \param image Imagem a ser corrigida
# \return matriz com os valores corrigidos
def equalize_hist(image):
    if not(isGray(image)):
        r, g, b = cv2.split(image)
        equalized_r = cv2.equalizeHist(r)
        equalized_g = cv2.equalizeHist(g)
        equalized_b = cv2.equalizeHist(b)
        return cv2.merge([equalized_r, equalized_g, equalized_b])
    else:
        return cv2.equalizeHist(image)

# \brief Funçao que aplica o filtro Laplaciano de acordo com a opçao passada
# 
# \param image Imagem a ser filtrada
# \param type Tipo do kernel a ser aplicado 0-[+-4] 1-[+-8]
# \return Matriz com a imagem após a borda ser adicionada 
def sharpening(image, type):
    kernel_8 = np.array([[1,1,1], [1, -8, 1], [1, 1, 1]])
    kernel_4 = np.array([[0,1,0], [1, -4, 1], [0, 1, 0]])
    if type == 0:
        laplacian =  cv2.filter2D(image, -1, kernel_4)
        return cv2.subtract(image, laplacian)
    elif type == 1:
        laplacian =  cv2.filter2D(image, -1, kernel_8)
        return cv2.subtract(image, laplacian)
    else:
        print('Tipo laplaciano inválido')
        return False
    
def Prog1():
    # manual resize
    original_image = cv2.imread('images/test80.jpg')
    my_resized = dec_int(original_image)
    # opencv resize
    compressed = cv2.resize(original_image, (int(original_image.shape[1]/2), int(original_image.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
    interpolated = cv2.resize(compressed, (int(compressed.shape[1]*2), int(compressed.shape[0]*2)), interpolation = cv2.INTER_CUBIC)
    # Descomentar duas linhas abaixo para ver o redimensionamento do opencv
    # cv2.imshow('OpenCV reduce', compressed)
    # cv2.imshow('OpenCV interpolated', interpolated)
    my_resized_improved = edge_improv(my_resized)
    opencv_resized_improved = edge_improv(interpolated)
    cv2.imshow('My resize improved', my_resized_improved)
    cv2.imshow('OpenCV resize improved', opencv_resized_improved)

def Prog2(): 
    # car.png manipulations
    car = cv2.imread('images/car.png')
    cv2.imshow('Car Original', car)
    power_law(car, 0.3)
    power_law(car, 0.6)
    power_law(car, 0.9)
    cv2.waitKey()
    cv2.destroyAllWindows()
    power_law(car, 2)
    power_law(car, 3)
    power_law(car, 4)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # crowd.png manipulations
    crowd = cv2.imread('images/crowd.png')
    cv2.imshow('Crowd Original', crowd)
    power_law(crowd, 0.3)
    power_law(crowd, 0.6)
    power_law(crowd, 0.9)
    cv2.waitKey()
    cv2.destroyAllWindows()
    power_law(crowd, 2)
    power_law(crowd, 3)
    power_law(crowd, 4)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # university.png manipulations
    university = cv2.imread('images/university.png')
    cv2.imshow('University Original', university)
    power_law(university, 0.3)
    power_law(university, 0.6)
    power_law(university, 0.9)
    cv2.waitKey()
    cv2.destroyAllWindows()
    power_law(university, 2)
    power_law(university, 3)
    power_law(university, 4)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Equalizando histograma da imagem car.png
    equalized_car = equalize_hist(car)
    equalized_crowd = equalize_hist(crowd)
    equalized_university = equalize_hist(university)
    cv2.imshow('Equalized car', equalized_car)
    cv2.imshow('Equalized crowd', equalized_crowd)
    cv2.imshow('Equalized university', equalized_university)
    cv2.waitKey()   
    cv2.destroyAllWindows()

    # Preparando e plotando histogramas e funções de distribuição acumulada
    fig, ax = plt.subplots(2,2)
    plt.tight_layout()
    fig.set_size_inches(10,7)
    grayscale_car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    original_flat = grayscale_car.flatten()
    ax[0][0].hist(original_flat, bins=255, range=[0,255])
    ax[0][0].set_title('Histograma não equalizado da imagem car.png')
    count, number = np.histogram(original_flat, bins=255)
    ax[0][1].plot(np.cumsum(count/len(original_flat)))
    ax[0][1].set_title('CDF da imagem não equalizada')
    grayscale_car_equalized = cv2.cvtColor(equalized_car, cv2.COLOR_BGR2GRAY)
    flat = grayscale_car_equalized.flatten()
    ax[1][0].hist(flat, bins=255)
    ax[1][0].set_title('Histograma equalizado da imagem car.png')
    count, number = np.histogram(flat, bins=255)
    ax[1][1].plot(np.cumsum(count/len(flat)))
    ax[1][1].set_title('CDF da imagem equalizada')
    plt.show()

def Prog3():
    image = cv2.imread('images/Image1.pgm')
    # 2.1
    first = sharpening(image, 1)
    cv2.imshow('Imagem original', image)
    cv2.imshow('Laplaciano todas as direcoes', first)
    # 2.2
    first_gaussian = cv2.GaussianBlur(image, [3,3], 0.5)
    second = sharpening(first_gaussian, 0)
    cv2.imshow('Laplaciano com gaussiano 0.5', second)
    # 2.3
    second_gaussian = cv2.GaussianBlur(image, [3,3], 1)
    third = sharpening(second_gaussian, 0)
    cv2.imshow('Laplaciano com gaussiano 1', third)
 
# Para prosseguir nos itens pedidos no trabalho
# basta apertar qualquer tecla
Prog1()
cv2.waitKey()
cv2.destroyAllWindows()

Prog2()
cv2.waitKey()

Prog3()
cv2.waitKey()
