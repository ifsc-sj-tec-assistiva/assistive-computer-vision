# Assistive Computer Vision (Tecnologia Assistiva)

Solução de acessibilidade e tecnologia assistiva baseada em Visão Computacional e Inteligência Artificial, projetada para permitir o controle do computador e a interação com o sistema operacional sem a necessidade de periféricos físicos convencionais.

O projeto utiliza o MediaPipe Tasks API para rastreamento biométrico de alta precisão e o PyAutoGUI para mapeamento direto no ambiente do usuário.

## Autores e Responsáveis

* Eduardo Cardoso Oliveira
* Victor Hugo Andrade

## Funcionalidades

* Controlo do Mouse por Rastreamento Facial (detector_rosto.py):
  * Mapeamento contínuo em tempo real com MediaPipe Tasks e baixa latência.
  * Movimentação suave e ergonômica com filtro EMA (anti-tremor) e sensibilidade ajustável.
  * **Calibração de Centro (Tecla C):** Adapta a posição de repouso confortável do usuário ao centro da tela.
  * **Dwell Click (Tecla D):** Clique automático por permanência com anel de carregamento visual ao redor da mira.
  * **Ações por Blendshapes Faciais:**
    * Piscada esquerda: Clique com o botão esquerdo.
    * Piscada direita: Clique com o botão direito.
    * Boca aberta: Modo Rolagem suave de páginas (Scroll).
    * Sobrancelhas levantadas (ou Tecla P): Pausa / Retoma o rastreamento.
  * **Arrastar e Soltar (Tecla A):** Permite mover janelas e selecionar textos.
  * **Teclado Virtual (Tecla T):** Aciona o teclado em tela nativo do Windows.
  * **Feedback Sonoro Sutil:** Bipes discretos em segundo plano para confirmação de ações.

* Controle por Gestos Manuais (detector_maos.py):
  * Rastreamento e reconhecimento das articulações da mão em tempo real para comandos complementares.

---

## Tecnologias Utilizadas

* Python 3.10+
* OpenCV (opencv-python): Captura de vídeo da webcam e renderização de elementos visuais.
* MediaPipe (mediapipe): Modelos pré-treinados de aprendizado profundo para detecção de malha facial e mãos via CPU.
* PyAutoGUI (pyautogui): Automação e envio de instruções de controle ao sistema operacional.

## Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone [[https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/ifsc-sj-tec-assistiva/assistive-computer-vision.git)
cd assistive-computer-vision

```

### 2. Configurar o Ambiente Virtual

Recomenda-se o uso de um ambiente virtual para isolar as dependências do projeto:

No Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

No Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt

```

---

## Modelos Pré-treinados (MediaPipe Tasks)

Devido ao tamanho dos arquivos binários, os modelos pré-treinados não são enviados diretamente no repositório. Baixe-os para a pasta raiz do projeto antes de executar:

### No Windows (PowerShell):

```powershell
# Modelo Facial
Invoke-WebRequest -Uri "[https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)" -OutFile "face_landmarker.task"

# Modelo Manual
Invoke-WebRequest -Uri "[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)" -OutFile "hand_landmarker.task"

```

### No Linux/macOS (wget ou curl):

```bash
wget [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
wget [https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

```

---

## Como Executar

### Rastreamento Facial:

```bash
python detector_rosto.py

```

### Rastreamento de Mãos:

```bash
python detector_maos.py

```

Nota de Utilização: Para encerrar a execução do programa a qualquer momento, pressione a tecla ESC na janela de vídeo ativa.

