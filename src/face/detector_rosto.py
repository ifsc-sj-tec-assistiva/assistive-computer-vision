import cv2
import os
import sys
import math
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

pyautogui.FAILSAFE = False

class DetectorRosto:
    def __init__(self, max_rostos=1, deteccao_confianca=0.5, rastreio_confianca=0.5):
        self.max_rostos = max_rostos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.resultados = None 
        
        self.tela_largura, self.tela_altura = pyautogui.size()
        
        caminho_modelo = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        if not os.path.exists(caminho_modelo):
            print(f"Erro: O ficheiro de modelo '{caminho_modelo}' não foi encontrado.")
            sys.exit(1)

        base_options = python.BaseOptions(model_asset_path=caminho_modelo)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_rostos,
            min_face_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def calcular_distancia(self, ponto1, ponto2):
        return math.hypot(ponto2[0] - ponto1[0], ponto2[1] - ponto1[1])

    def encontrar_rostos(self, imagem, desenho=True):
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        
        self.resultados = self.detector.detect(mp_image)
        
        if self.resultados and self.resultados.face_landmarks and desenho:
            pontos_focais = {
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                300, 293, 334, 296, 336, 285, 295, 282, 283, 276
            }
            
            for rosto_pontos in self.resultados.face_landmarks:
                altura, largura, _ = imagem.shape
                for idx, ponto in enumerate(rosto_pontos):
                    if idx in pontos_focais:
                        px = int(ponto.x * largura)
                        py = int(ponto.y * altura)
                        cv2.circle(imagem, (px, py), 1, (0, 255, 0), cv2.FILLED)
                        
        return imagem

    def controlar_mouse(self, imagem):
        if not self.resultados or not self.resultados.face_landmarks:
            return imagem
            
        altura_img, largura_img, _ = imagem.shape
        rosto_pontos = self.resultados.face_landmarks[0] 
        
        nariz = rosto_pontos[1]
        nariz_x_px = int(nariz.x * largura_img)
        nariz_y_px = int(nariz.y * altura_img)
        
        mouse_x = int(nariz.x * self.tela_largura)
        mouse_y = int(nariz.y * self.tela_altura)
        
        pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

        olho_esq_topo = (int(rosto_pontos[159].x * largura_img), int(rosto_pontos[159].y * altura_img))
        olho_esq_base = (int(rosto_pontos[145].x * largura_img), int(rosto_pontos[145].y * altura_img))
        
        distancia = self.calcular_distancia(olho_esq_topo, olho_esq_base)
        
        if distancia < 6.0: 
            pyautogui.click()
            pyautogui.sleep(0.3)
            
        cv2.circle(imagem, (nariz_x_px, nariz_y_px), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(imagem, olho_esq_topo, olho_esq_base, (0, 255, 255), 2)
        
        return imagem

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível aceder à webcam.")
        return

    detector = DetectorRosto()
    
    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            break
            
        imagem = cv2.flip(imagem, 1)
        imagem = detector.encontrar_rostos(imagem, desenho=True)
        imagem = detector.controlar_mouse(imagem)
        
        cv2.imshow("Tecnologia Assistiva - Controlo Facial", imagem)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()