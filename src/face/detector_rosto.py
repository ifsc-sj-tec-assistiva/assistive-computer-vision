import cv2
import os
import sys
import time
import math
import threading
import subprocess
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

try:
    import winsound
except ImportError:
    winsound = None

# Desativa pausas internas do PyAutoGUI para latência zero
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

def emitir_feedback_sonoro(tipo="click"):
    """Emite bipes sonoros discretos em thread separada para não travar o loop visual."""
    if not winsound:
        return

    def _tocar():
        try:
            if tipo == "click":
                winsound.Beep(1200, 45)
            elif tipo == "right_click":
                winsound.Beep(900, 60)
            elif tipo == "dwell":
                winsound.Beep(1400, 70)
            elif tipo == "calibrate":
                winsound.Beep(1600, 90)
            elif tipo == "pause":
                winsound.Beep(500, 80)
            elif tipo == "unpause":
                winsound.Beep(1000, 80)
            elif tipo == "drag_start":
                winsound.Beep(750, 100)
            elif tipo == "drag_end":
                winsound.Beep(1100, 100)
        except Exception:
            pass

    threading.Thread(target=_tocar, daemon=True).start()

class DetectorRosto:
    def __init__(
        self,
        max_rostos=1,
        deteccao_confianca=0.5,
        rastreio_confianca=0.5,
        sensibilidade_x=2.2,
        sensibilidade_y=2.4,
        suavizacao=0.25,
        tempo_dwell=1.0,         # Segundos imóvel para disparar o clique automático
        raio_tolerancia_dwell=25  # Tolerância em pixels para considerar "imóvel"
    ):
        self.max_rostos = max_rostos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.sensibilidade_x = sensibilidade_x
        self.sensibilidade_y = sensibilidade_y
        self.suavizacao = suavizacao
        
        # Dimensões do monitor
        self.tela_largura, self.tela_altura = pyautogui.size()
        
        # Posição anterior (para suavização EMA)
        self.mouse_antigo_x = self.tela_largura / 2
        self.mouse_antigo_y = self.tela_altura / 2
        
        # Calibração da postura de repouso (Zero / Tare)
        self.centro_repouso_x = 0.5
        self.centro_repouso_y = 0.5
        self.calibrado = False

        # Estados de operação
        self.pausado = False
        self.arrastando = False
        self.dwell_ativado = True
        
        # Dwell Click (Clique por permanência)
        self.tempo_dwell = tempo_dwell
        self.raio_tolerancia_dwell = raio_tolerancia_dwell
        self.dwell_pos_referencia = None
        self.dwell_tempo_inicio = None
        self.dwell_progresso = 0.0
        self.dwell_cooldown_fim = 0.0
        
        # Gestos e temporizadores (debounce não-bloqueante)
        self.ultimo_timestamp_ms = 0
        self.ultimo_clique_esq = 0.0
        self.ultimo_clique_dir = 0.0
        self.ultimo_toggle_pausa = 0.0
        self.cooldown_clique = 0.4
        self.tempo_inicio_fechado_esq = 0.0
        self.tempo_inicio_fechado_dir = 0.0
        self.y_referencia_rolagem = None
        
        self.resultados = None

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

    def calibrar_centro(self, nariz_x=None, nariz_y=None):
        """Define a postura atual da cabeça como o centro da tela (Repouso)."""
        if nariz_x is not None and nariz_y is not None:
            self.centro_repouso_x = nariz_x
            self.centro_repouso_y = nariz_y
            self.calibrado = True
            emitir_feedback_sonoro("calibrate")
            print(f"[Calibração] Novo centro de repouso: ({nariz_x:.3f}, {nariz_y:.3f})")

    def toggle_pausa(self):
        """Alterna entre pausado e ativo."""
        self.pausado = not self.pausado
        if self.arrastando:
            pyautogui.mouseUp()
            self.arrastando = False
        emitir_feedback_sonoro("pause" if self.pausado else "unpause")

    def toggle_dwell(self):
        """Liga/Desliga o Dwell Click (clique automático ao ficar imóvel)."""
        self.dwell_ativado = not self.dwell_ativado
        self.dwell_tempo_inicio = None
        self.dwell_progresso = 0.0
        emitir_feedback_sonoro("click")

    def toggle_arrastar(self):
        """Alterna estado de Arrastar e Soltar (MouseDown / MouseUp)."""
        self.arrastando = not self.arrastando
        if self.arrastando:
            pyautogui.mouseDown()
            emitir_feedback_sonoro("drag_start")
        else:
            pyautogui.mouseUp()
            emitir_feedback_sonoro("drag_end")

    def abrir_teclado_virtual(self):
        """Abre o Teclado em Tela nativo do Windows."""
        try:
            subprocess.Popen("osk.exe", shell=True)
            emitir_feedback_sonoro("click")
        except Exception as e:
            print(f"Erro ao abrir Teclado Virtual: {e}")

    def extrair_blendshapes(self):
        """Retorna dicionário com os blendshapes faciais detectados."""
        if not self.resultados or not self.resultados.face_blendshapes:
            return {}
        return {cat.category_name: cat.score for cat in self.resultados.face_blendshapes[0]}

    def processar_frame(self, imagem):
        """Executa a inferência contínua assíncrona do MediaPipe Tasks."""
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
        """Processa a interação com o cursor, cliques, gestos e HUD."""
        if not self.resultados or not self.resultados.face_landmarks:
            return imagem

        altura_img, largura_img, _ = imagem.shape
        rosto_pontos = self.resultados.face_landmarks[0]
        agora = time.time()

        # Coordenada do nariz (referência de mira)
        nariz = rosto_pontos[1]
        nariz_px_x = int(nariz.x * largura_img)
        nariz_px_y = int(nariz.y * altura_img)

        # Calibração automática no primeiro frame detectado
        if not self.calibrado:
            self.calibrar_centro(nariz.x, nariz.y)

        # Mapeamento relativo à postura de repouso calibrada
        delta_x = nariz.x - self.centro_repouso_x
        delta_y = nariz.y - self.centro_repouso_y

        # Projeção na tela do monitor com ganho de sensibilidade
        alvo_mouse_x = (0.5 + (delta_x * self.sensibilidade_x)) * self.tela_largura
        alvo_mouse_y = (0.5 + (delta_y * self.sensibilidade_y)) * self.tela_altura

        # Limitar às bordas do monitor
        alvo_mouse_x = max(0, min(self.tela_largura - 1, alvo_mouse_x))
        alvo_mouse_y = max(0, min(self.tela_altura - 1, alvo_mouse_y))

        suave_x, suave_y = self.suavizar_coordenada(alvo_mouse_x, alvo_mouse_y)

        # Leitura dos blendshapes faciais
        blendshapes = self.extrair_blendshapes()
        piscar_esq = blendshapes.get("eyeBlinkLeft", 0.0)
        piscar_dir = blendshapes.get("eyeBlinkRight", 0.0)
        boca_aberta = blendshapes.get("jawOpen", 0.0)
        sobrancelha_alta = blendshapes.get("browInnerUp", 0.0)

        # Alternar Pausa com sobrancelhas levantadas (> 0.65 por 0.5s)
        if sobrancelha_alta > 0.65 and (agora - self.ultimo_toggle_pausa > 1.2):
            self.toggle_pausa()
            self.ultimo_toggle_pausa = agora

        status_acao = "PAUSADO" if self.pausado else ("ARRASTANDO" if self.arrastando else "LIVRE")

        if not self.pausado:
            # 1. Modo Rolagem (Boca Aberta)
            if boca_aberta > 0.45:
                status_acao = "ROLAGEM"
                if self.y_referencia_rolagem is None:
                    self.y_referencia_rolagem = nariz.y
                else:
                    diff_y = self.y_referencia_rolagem - nariz.y
                    if abs(diff_y) > 0.015:
                        scroll_amount = int(diff_y * 140)
                        pyautogui.scroll(scroll_amount)
                # Reseta dwell ao rolar
                self.dwell_tempo_inicio = None
                self.dwell_progresso = 0.0
            else:
                self.y_referencia_rolagem = None
                # Movimenta o cursor instantaneamente sem delay
                pyautogui.moveTo(suave_x, suave_y)

                # 2. Dwell Click (Clique por Permanência)
                if self.dwell_ativado and (agora > self.dwell_cooldown_fim):
                    if self.dwell_pos_referencia is None:
                        self.dwell_pos_referencia = (suave_x, suave_y)
                        self.dwell_tempo_inicio = agora
                        self.dwell_progresso = 0.0
                    else:
                        dist = math.hypot(suave_x - self.dwell_pos_referencia[0], suave_y - self.dwell_pos_referencia[1])
                        if dist < self.raio_tolerancia_dwell:
                            tempo_decorrido = agora - self.dwell_tempo_inicio
                            self.dwell_progresso = min(1.0, tempo_decorrido / self.tempo_dwell)
                            if tempo_decorrido >= self.tempo_dwell:
                                pyautogui.click()
                                emitir_feedback_sonoro("dwell")
                                status_acao = "DWELL CLIQUE"
                                self.dwell_tempo_inicio = None
                                self.dwell_pos_referencia = None
                                self.dwell_progresso = 0.0
                                self.dwell_cooldown_fim = agora + 0.6
                        else:
                            # Se moveu além do raio, reinicia a contagem na nova posição
                            self.dwell_pos_referencia = (suave_x, suave_y)
                            self.dwell_tempo_inicio = agora
                            self.dwell_progresso = 0.0
                else:
                    self.dwell_progresso = 0.0

            # 3. Clique Esquerdo por Piscada do Olho Esquerdo
            if piscar_esq > 0.55 and piscar_dir < 0.40:
                if self.tempo_inicio_fechado_esq == 0.0:
                    self.tempo_inicio_fechado_esq = agora
                elif (agora - self.tempo_inicio_fechado_esq >= 0.15) and (agora - self.ultimo_clique_esq > self.cooldown_clique):
                    pyautogui.click(button="left")
                    emitir_feedback_sonoro("click")
                    self.ultimo_clique_esq = agora
                    self.tempo_inicio_fechado_esq = 0.0
                    self.dwell_cooldown_fim = agora + 0.5
                    status_acao = "CLIQUE ESQ"
            else:
                self.tempo_inicio_fechado_esq = 0.0

            # 4. Clique Direito por Piscada do Olho Direito
            if piscar_dir > 0.55 and piscar_esq < 0.40:
                if self.tempo_inicio_fechado_dir == 0.0:
                    self.tempo_inicio_fechado_dir = agora
                elif (agora - self.tempo_inicio_fechado_dir >= 0.15) and (agora - self.ultimo_clique_dir > self.cooldown_clique):
                    pyautogui.click(button="right")
                    emitir_feedback_sonoro("right_click")
                    self.ultimo_clique_dir = agora
                    self.tempo_inicio_fechado_dir = 0.0
                    self.dwell_cooldown_fim = agora + 0.5
                    status_acao = "CLIQUE DIR"
            else:
                self.tempo_inicio_fechado_dir = 0.0

        # Renderização do HUD visual de Tecnologia Assistiva
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

        # Ponto de repouso calibrado desenhado na imagem
        cx_px = int(self.centro_repouso_x * w)
        cy_px = int(self.centro_repouso_y * h)
        cv2.drawMarker(imagem, (cx_px, cy_px), (80, 80, 80), cv2.MARKER_CROSS, 20, 1)

        # Mira na ponta do nariz com anel de Dwell Click dinâmico
        cor_mira = (0, 0, 255) if self.pausado else ((0, 140, 255) if self.arrastando else (0, 255, 0))
        cv2.circle(imagem, (nx, ny), 5, cor_mira, -1)

        # Se Dwell estiver ativo e carregando, desenha um anel de progresso radial
        raio_anel = 18
        cv2.circle(imagem, (nx, ny), raio_anel, (80, 80, 80), 1, cv2.LINE_AA)
        if self.dwell_progresso > 0.0 and not self.pausado:
            angulo_fim = int(self.dwell_progresso * 360)
            cv2.ellipse(
                imagem,
                (nx, ny),
                (raio_anel, raio_anel),
                -90,
                0,
                angulo_fim,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

        # Painel Informativo Lateral Translúcido
        overlay = imagem.copy()
        cv2.rectangle(overlay, (10, 10), (330, 200), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, imagem, 0.25, 0, imagem)

        # Status principal
        cor_status = (0, 0, 255) if self.pausado else (0, 255, 128)
        cv2.putText(imagem, f"STATUS: {acao}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)

        # Barras de nível de gestos
        self._desenhar_barra(imagem, "Olho Esq (Esq)", 20, 48, p_esq, 0.55)
        self._desenhar_barra(imagem, "Olho Dir (Dir)", 20, 66, p_dir, 0.55)
        self._desenhar_barra(imagem, "Boca (Scroll)", 20, 84, boca, 0.45)
        self._desenhar_barra(imagem, "Sobranc (Pausa)", 20, 102, sobrancelha, 0.65)

        # Atalhos rápidos e recursos ativos
        dwell_txt = "ON" if self.dwell_ativado else "OFF"
        cv2.putText(imagem, f"[D] Dwell Click: {dwell_txt}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        drag_txt = "ATIVO" if self.arrastando else "INATIVO"
        cv2.putText(imagem, f"[A] Arrastar: {drag_txt}", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(imagem, "[C] Calibrar Centro (Zero)", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(imagem, "[T] Teclado Virtual | [P] Pausar", (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(imagem, "[ESC] Sair", (20, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 255), 1)

        return imagem

    def _desenhar_barra(self, imagem, label, x, y, valor, limite):
        largura_barra = 100
        preenchimento = int(np.clip(valor, 0.0, 1.0) * largura_barra)
        cor = (0, 255, 0) if valor >= limite else (160, 160, 160)

        cv2.putText(imagem, label, (x, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1)
        cv2.rectangle(imagem, (x + 115, y), (x + 115 + largura_barra, y + 8), (45, 45, 45), -1)
        cv2.rectangle(imagem, (x + 115, y), (x + 115 + preenchimento, y + 8), cor, -1)
        lim_x = int(x + 115 + (limite * largura_barra))
        cv2.line(imagem, (lim_x, y - 2), (lim_x, y + 10), (0, 0, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    detector = DetectorRosto()
    tempo_anterior = time.time()

    print("\n==============================================")
    print("      TECNOLOGIA ASSISTIVA - CONTROLE FACIAL  ")
    print("==============================================")
    print("Comandos e Atalhos:")
    print(" [C] Calibrar Centro: Define sua postura confortável como centro da tela")
    print(" [D] Dwell Click: Liga/Desliga clique automático por permanência")
    print(" [A] Arrastar/Soltar: Ativa/desativa arrastar janelas/arquivos")
    print(" [T] Teclado Virtual: Abre o teclado de tela do Windows")
    print(" [P] Pausar / Retomar: Trava ou libera o cursor")
    print(" [ESC] Sair da aplicação\n")
    print("Gestos Faciais:")
    print(" - Olho Esquerdo fechado: Clique esquerdo")
    print(" - Olho Direito fechado: Clique direito")
    print(" - Boca aberta: Modo Rolagem (Scroll suave)")
    print(" - Sobrancelhas levantadas: Pausa / Retoma")
    print("==============================================\n")

    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            break

        imagem = cv2.flip(imagem, 1)

        # Inferência com MediaPipe Tasks
        detector.processar_frame(imagem)

        # Controle e HUD
        imagem = detector.controlar(imagem)

        # FPS em tempo real
        tempo_atual = time.time()
        fps = 1.0 / (tempo_atual - tempo_anterior) if (tempo_atual - tempo_anterior) > 0 else 0
        tempo_anterior = tempo_atual
        cv2.putText(
            imagem,
            f"FPS: {int(fps)}",
            (imagem.shape[1] - 105, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Tecnologia Assistiva - Controle Facial", imagem)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        elif tecla in (ord('c'), ord('C')):
            if detector.resultados and detector.resultados.face_landmarks:
                nariz = detector.resultados.face_landmarks[0][1]
                detector.calibrar_centro(nariz.x, nariz.y)
        elif tecla in (ord('p'), ord('P')):
            detector.toggle_pausa()
        elif tecla in (ord('d'), ord('D')):
            detector.toggle_dwell()
        elif tecla in (ord('a'), ord('A')):
            detector.toggle_arrastar()
        elif tecla in (ord('t'), ord('T')):
            detector.abrir_teclado_virtual()

    if detector.arrastando:
        pyautogui.mouseUp()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
