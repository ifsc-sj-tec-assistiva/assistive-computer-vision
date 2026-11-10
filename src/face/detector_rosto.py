import cv2
import os
import sys
import time
import math
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# Desativa pausas internas que causam travamento extremo no loop de vídeo
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

class DetectorRosto:
    def __init__(
        self,
        max_rostos=1,
        deteccao_confianca=0.5,
        rastreio_confianca=0.5,
        sensibilidade_x=1.8,
        sensibilidade_y=1.8,
        suavizacao=0.25,      # Fator de suavização EMA (0.1 = muito suave, 1.0 = instantâneo)
        margem_tela_x=0.20,   # Margem virtual da câmera (evita ter que virar muito o pescoço)
        margem_tela_y=0.20
    ):
        self.max_rostos = max_rostos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.sensibilidade_x = sensibilidade_x
        self.sensibilidade_y = sensibilidade_y
        self.suavizacao = suavizacao
        self.margem_x = margem_tela_x
        self.margem_y = margem_tela_y
        
        self.resultados = None
        self.ultimo_timestamp_ms = 0
        
        # Dimensões da tela do monitor
        self.tela_largura, self.tela_altura = pyautogui.size()
        
        # Posição anterior do mouse (para suavização EMA)
        self.mouse_antigo_x = self.tela_largura / 2
        self.mouse_antigo_y = self.tela_altura / 2
        
        # Estados de controle
        self.pausado = False
        self.modo_rolagem = False
        
        # Temporizadores de debounce para gestos (em segundos, não bloqueantes)
        self.ultimo_clique_esq = 0.0
        self.ultimo_clique_dir = 0.0
        self.ultimo_toggle_pausa = 0.0
        self.cooldown_clique = 0.35
        self.tempo_inicio_fechado_esq = 0.0
        self.tempo_inicio_fechado_dir = 0.0
        
        # Ponto de referência para rolagem
        self.y_referencia_rolagem = None

        # Carregamento do modelo MediaPipe Tasks
        caminho_modelo = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        if not os.path.exists(caminho_modelo):
            print(f"Erro: O arquivo de modelo '{caminho_modelo}' não foi encontrado.")
            sys.exit(1)

        base_options = python.BaseOptions(model_asset_path=caminho_modelo)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=self.max_rostos,
            min_face_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extrair_blendshapes(self):
        """Retorna um dicionário com os blendshapes faciais detectados."""
        if not self.resultados or not self.resultados.face_blendshapes:
            return {}
        
        blendshapes = {}
        for category in self.resultados.face_blendshapes[0]:
            blendshapes[category.category_name] = category.score
        return blendshapes

    def processar_frame(self, imagem):
        """Executa a inferência de forma assíncrona/temporal contínua."""
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.ultimo_timestamp_ms:
            timestamp_ms = self.ultimo_timestamp_ms + 1
        self.ultimo_timestamp_ms = timestamp_ms

        self.resultados = self.detector.detect_for_video(mp_image, timestamp_ms)
        return self.resultados

    def suavizar_coordenada(self, novo_x, novo_y):
        """Aplica filtro de Média Móvel Exponencial (EMA) para eliminar tremores."""
        suave_x = (self.suavizacao * novo_x) + ((1.0 - self.suavizacao) * self.mouse_antigo_x)
        suave_y = (self.suavizacao * novo_y) + ((1.0 - self.suavizacao) * self.mouse_antigo_y)
        self.mouse_antigo_x = suave_x
        self.mouse_antigo_y = suave_y
        return int(suave_x), int(suave_y)

    def controlar(self, imagem):
        """Processa a interação com o cursor, cliques e HUD na tela."""
        if not self.resultados or not self.resultados.face_landmarks:
            return imagem

        altura_img, largura_img, _ = imagem.shape
        rosto_pontos = self.resultados.face_landmarks[0]
        agora = time.time()

        # 1. Obtenção do ponto de rastreio (nariz - landmark 1 ou 4)
        nariz = rosto_pontos[1]
        nariz_px_x = int(nariz.x * largura_img)
        nariz_px_y = int(nariz.y * altura_img)

        # 2. Mapeamento ergonômico com zona de sensibilidade (Deadzone / Bounding Box)
        # O usuário não precisa virar o pescoço todo para cobrir a tela inteira
        min_x = self.margem_x
        max_x = 1.0 - self.margem_x
        min_y = self.margem_y
        max_y = 1.0 - self.margem_y

        norm_x = np.clip((nariz.x - min_x) / (max_x - min_x), 0.0, 1.0)
        norm_y = np.clip((nariz.y - min_y) / (max_y - min_y), 0.0, 1.0)

        alvo_mouse_x = norm_x * self.tela_largura
        alvo_mouse_y = norm_y * self.tela_altura

        suave_mouse_x, suave_mouse_y = self.suavizar_coordenada(alvo_mouse_x, alvo_mouse_y)

        # 3. Leitura dos blendshapes faciais
        blendshapes = self.extrair_blendshapes()
        piscar_esq = blendshapes.get("eyeBlinkLeft", 0.0)
        piscar_dir = blendshapes.get("eyeBlinkRight", 0.0)
        boca_aberta = blendshapes.get("jawOpen", 0.0)
        sobrancelha_alta = blendshapes.get("browInnerUp", 0.0)

        # Alternar Pausa (Sobrancelhas levantadas por 0.5s ou tecla 'p')
        if sobrancelha_alta > 0.60 and (agora - self.ultimo_toggle_pausa > 1.0):
            self.pausado = not self.pausado
            self.ultimo_toggle_pausa = agora

        # 4. Ações do Mouse (se não estiver pausado)
        status_acao = "LIVRE"
        if not self.pausado:
            # Modo Rolagem ativado quando a boca está aberta (> 0.45)
            if boca_aberta > 0.45:
                status_acao = "ROLAGEM"
                if self.y_referencia_rolagem is None:
                    self.y_referencia_rolagem = nariz.y
                else:
                    diff_y = self.y_referencia_rolagem - nariz.y
                    if abs(diff_y) > 0.02:
                        scroll_amount = int(diff_y * 120)
                        pyautogui.scroll(scroll_amount)
            else:
                self.y_referencia_rolagem = None
                # Movimentação instantânea sem duration (zero lag)
                pyautogui.moveTo(suave_mouse_x, suave_mouse_y)

            # Clique com botão esquerdo: Olho esquerdo fechado (> 0.55) e olho direito aberto
            if piscar_esq > 0.55 and piscar_dir < 0.40:
                if self.tempo_inicio_fechado_esq == 0.0:
                    self.tempo_inicio_fechado_esq = agora
                elif (agora - self.tempo_inicio_fechado_esq >= 0.15) and (agora - self.ultimo_clique_esq > self.cooldown_clique):
                    pyautogui.click(button="left")
                    self.ultimo_clique_esq = agora
                    self.tempo_inicio_fechado_esq = 0.0
                    status_acao = "CLIQUE ESQ"
            else:
                self.tempo_inicio_fechado_esq = 0.0

            # Clique com botão direito: Olho direito fechado (> 0.55) e olho esquerdo aberto
            if piscar_dir > 0.55 and piscar_esq < 0.40:
                if self.tempo_inicio_fechado_dir == 0.0:
                    self.tempo_inicio_fechado_dir = agora
                elif (agora - self.tempo_inicio_fechado_dir >= 0.15) and (agora - self.ultimo_clique_dir > self.cooldown_clique):
                    pyautogui.click(button="right")
                    self.ultimo_clique_dir = agora
                    self.tempo_inicio_fechado_dir = 0.0
                    status_acao = "CLIQUE DIR"
            else:
                self.tempo_inicio_fechado_dir = 0.0

        # 5. Renderização do HUD visual informativo e ergonômico
        imagem = self.desenhar_hud(
            imagem,
            nariz_px_x,
            nariz_px_y,
            piscar_esq,
            piscar_dir,
            boca_aberta,
            sobrancelha_alta,
            status_acao
        )

        return imagem

    def desenhar_hud(self, imagem, nx, ny, p_esq, p_dir, boca, sobrancelha, acao):
        h, w, _ = imagem.shape

        # Retângulo da área de sensibilidade da cabeça (ergonomia)
        x1 = int(self.margem_x * w)
        y1 = int(self.margem_y * h)
        x2 = int((1.0 - self.margem_x) * w)
        y2 = int((1.0 - self.margem_y) * h)
        cv2.rectangle(imagem, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)

        # Ponto do nariz (mira do cursor)
        cor_mira = (0, 0, 255) if self.pausado else (0, 255, 0)
        cv2.circle(imagem, (nx, ny), 6, cor_mira, -1)
        cv2.circle(imagem, (nx, ny), 12, cor_mira, 1)

        # Painel superior translúcido para informações
        overlay = imagem.copy()
        cv2.rectangle(overlay, (10, 10), (330, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, imagem, 0.3, 0, imagem)

        # Status do Rastreamento
        status_txt = "PAUSADO [P]" if self.pausado else f"ATIVO ({acao})"
        cor_status = (0, 0, 255) if self.pausado else (0, 255, 128)
        cv2.putText(imagem, status_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_status, 2)

        # Barras de nível de gestos
        self._desenhar_barra(imagem, "Olho Esq (Esq)", 20, 55, p_esq, 0.55)
        self._desenhar_barra(imagem, "Olho Dir (Dir)", 20, 75, p_dir, 0.55)
        self._desenhar_barra(imagem, "Boca (Scroll)", 20, 95, boca, 0.45)
        self._desenhar_barra(imagem, "Sobranc (Pausa)", 20, 115, sobrancelha, 0.60)

        return imagem

    def _desenhar_barra(self, imagem, label, x, y, valor, limite):
        largura_barra = 100
        preenchimento = int(np.clip(valor, 0.0, 1.0) * largura_barra)
        cor = (0, 255, 0) if valor >= limite else (180, 180, 180)

        cv2.putText(imagem, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        # Fundo da barra
        cv2.rectangle(imagem, (x + 115, y), (x + 115 + largura_barra, y + 10), (50, 50, 50), -1)
        # Barra preenchida
        cv2.rectangle(imagem, (x + 115, y), (x + 115 + preenchimento, y + 10), cor, -1)
        # Marcador do limite
        lim_x = int(x + 115 + (limite * largura_barra))
        cv2.line(imagem, (lim_x, y - 2), (lim_x, y + 12), (0, 0, 255), 1)

def main():
    # Inicializa captura com resolução balanceada para alta taxa de quadros (FPS)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    detector = DetectorRosto()
    tempo_anterior = time.time()

    print("\n--- TECNOLOGIA ASSISTIVA: CONTROLE FACIAL ---")
    print("Controles:")
    print(" - Mover a cabeça: Move o cursor com suavização inteligente")
    print(" - Piscar olho esquerdo: Clique com botão esquerdo")
    print(" - Piscar olho direito: Clique com botão direito")
    print(" - Abrir a boca e mover cabeça: Modo Rolagem (Scroll)")
    print(" - Levantar sobrancelhas ou tecla 'P': Pausar / Retomar rastreamento")
    print(" - Tecla 'ESC': Encerrar programa\n")

    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            break

        # Espelhar para sensação natural de espelho
        imagem = cv2.flip(imagem, 1)

        # Inferência com MediaPipe Tasks
        detector.processar_frame(imagem)

        # Controle do mouse e renderização do HUD
        imagem = detector.controlar(imagem)

        # Cálculo de FPS real
        tempo_atual = time.time()
        fps = 1.0 / (tempo_atual - tempo_anterior) if (tempo_atual - tempo_anterior) > 0 else 0
        tempo_anterior = tempo_atual
        cv2.putText(
            imagem,
            f"FPS: {int(fps)}",
            (imagem.shape[1] - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Tecnologia Assistiva - Controle Facial", imagem)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        elif tecla in (ord('p'), ord('P')):
            detector.pausado = not detector.pausado

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
