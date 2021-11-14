import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

while True:
    print("""
		+--------------------------------------+
		|         Selecione uma Opção          |
		+--------------------------------------+

		+--------------------------------------+
		| [1] Escolher Imagem                  |
		+--------------------------------------+
		| [2] Sair do programa                 |
		+--------------------------------------+


	            """)
    opcao = int(input("Digite a opção escolhida: "))

    if (
            opcao == 1):  ###############################################################################################################################
        print(" ")
        print("ATENÇÃO: AS IMAGENS DEVEM CONTER O MESMO TAMANHO.")
        print("EXEMPLO: 600x600 & 600x600")
        print(" ")
        print("Digite o nome e a extensão da primeira imagem que deseja abrir para sobrepor.")
        print("Exemplo: imagem-1.jpg ")

        # carrega as imagens a serem comparadas
        read = input("Nome da imagem: ")
        imagem1 = cv2.imread(read)
        imagem2 = cv2.imread(read)  # seta tipo de váriavel (a imagem não é utilizada)

        print(" ")
        print("Agora digite o nome e a extensão da imagem a ser sobreposta.")
        print("Exemplo: imagem-2.jpg ")
        read2 = input("Nome da imagem: ")

        # seleção de imagem a partir da digitação no terminal
        imagem2 = cv2.imread(read2)
        altura2 = imagem2.shape[0]
        largura2 = imagem2.shape[1]

        # defini aqui para deixar a resolução da imagem padrão ao ser exibida na tela após os processamento
        imagem1 = cv2.resize(imagem1, (1500, 900)) # alturaxlargura
        imagem2 = cv2.resize(imagem2, (1500, 900)) # alturaxlargura

        # opção manual para escolher a dimensão da imagem (ultimo caso)
        # dim = (500, 500) #aqui eu posso definir a dimensão da imagem a partir da própria
        # imagem1 = cv2.resize(imagem1, dim, interpolation=cv2.INTER_AREA)
        # imagem2 = cv2.resize(imagem2, dim, interpolation=cv2.INTER_AREA)

        # Converter imagens em tons de cinza
        antes_do_cinza = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
        depois_do_cinza = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

        # Structural Similarity Index (Índice de Similaridade Estrutural)
        # Usando a função 'structural_similarity', calcula-se uma pontuação e a imagem da diferença
        # A pontuação representa o índice de similaridade estrutural entre as duas imagens de entrada.
        # Este valor pode cair no intervalo [-1, 1] com um valor de um sendo uma “correspondência perfeita”.
        (pontuacao, diferenca) = structural_similarity(antes_do_cinza, depois_do_cinza, full=True)

        # A diferença image contém as diferenças reais da imagem entre as duas imagens de entrada que desejamos visualizar.
        # A imagem de diferença é atualmente representada como um tipo de dados de ponto flutuante no intervalo [0, 1], portanto,
        # primeiro convertemos a matriz em inteiros sem sinal de 8 bits no intervalo [0, 255] antes de podermos processá-lo posteriormente usando OpenCV.
        diferenca = (diferenca * 255).astype(
            "uint8")  # copia a imagem para outro objeto de imagem, este método é útil quando precisamos copiar a imagem, mas também reter o original.

        # Limite a imagem de diferença, seguido por encontrar contornos para obter as regiões das duas imagens de entrada que diferem
        # Encontrar os contornos para colocar retângulos em torno das regiões identificadas como “diferentes”
        # THRESH_BINARY_INV e THRESH_OTSU usada obter detalhes sobre a configuração de limiar bimodal de Otsu
        thresh = cv2.threshold(diferenca, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contornos = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Os contornos inicializa com [0], depois serão armazenados na lista
        contornos = contornos[0] \
            if len(contornos) == 2 \
            else contornos[1]

        # mascara e pontos de pixel da imagem
        # a mascara retorna uma nova matriz de forma(imagem1) e tipo fornecidos, preenchida com zeros
        # a função Numpy(np) fornece coordenadas no formato (linha, coluna)
        mascara = np.zeros(imagem1.shape, dtype="uint8")
        preenchido_depois = imagem2.copy()  # copia a imagem para outro objeto de imagem, este método é útil quando precisamos copiar a imagem, mas também reter o original.

        # no for abaixo obtem-se a volta sobre os contornos onde,
        # calcula a caixa delimitadora do contorno e, em seguida, desenha a
        # caixa delimitadora em ambas as imagens de entrada para representar onde as duas imagens são diferentes
        numeroDiferencas = 0  # declaração de variavel para achar o número de diferenças na image,

        for c in contornos:

            # proporção entre a área do contorno e a área do retângulo delimitador
            area = cv2.contourArea(c)
            if area > 40:
                numeroDiferencas += 1
                # proporção entre a largura e a altura do retângulo delimitador do objeto
                x, y, w, h = cv2.boundingRect(c)

                # valores para desenhar um retângulo verde em cada imagem com
                cv2.rectangle(imagem1, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(imagem2, (x, y), (x + w, y + h), (36, 255, 12), 2)

                # valores para desenhar a mascara verde em cada imagem com
                cv2.drawContours(mascara, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(preenchido_depois, [c], 0, (0, 255, 0), -1)

        print("\n------------------INFORMAÇÕES------------------")
        print("Semelhança de imagem", pontuacao)  # impressão no console
        print("Tons de diferença na imagem = ", numeroDiferencas)  # impressão no console
        print("Tamanho da imagem: ", imagem1.shape)  # mostra informações a respeito da dimensão da imagem1
        print("Tamanho da imagem: ", imagem2.shape)  # mostra informações a respeito da dimensão da imagem2
        print("------------------------------------\n")

        resposta = input("Deseja visualizar a imagem agora? [S] [N] : ")
        if (resposta == "S" or resposta == "s"):
            # impressões das janelas
            # cv2.imshow("imagem1", imagem1)
            cv2.imshow("imagem2", imagem2)
            # cv2.imshow("diferenca", diferenca)
            # cv2.imshow("mascara", mascara)
            # cv2.imshow("preenchido imagem2", preenchido_depois)
            cv2.waitKey(0)

            print("Digite qualquer tecla para fechar.")

    elif (
            opcao == 2):  ###############################################################################################################################
        print("Finalizando o programa...")
        cv2.destroyAllWindows()
        sys.exit()

    else:
        print("Opção inserida é invalida, tente novamente!")
