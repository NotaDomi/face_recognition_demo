# Face Recognition Demo

Questa è una demo per il riconoscimento facciale usando un ___database vettoriale___.

## Setup

Segui questi passaggi per configurare l'ambiente ed eseguire la demo.

### 1. Installazione Prerequisiti

*   **Python:**   Si consiglia di gestire le versioni di Python usando `pyenv-win`.
    1.  Installa `pyenv-win` eseguendo:
        ```powershell
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
        ```
    2.  Dopo l'installazione è necessario riavviare la shell.
    3.  Elenca le versioni disponibili da installare:
        ```bash
        pyenv install -l
        ```
    4.  Installa la versione di Python desiderata:
        ```bash
        pyenv install <version>
        ```
    5.  Imposta la versione globale:
        ```bash
        pyenv global <version>
        ```
*   **CMake:**
    *   Scarica e installa CMake dal [sito ufficiale](https://cmake.org/download/). Assicurati di aggiungerlo al PATH di sistema durante l'installazione.
*   **Visual Studio C++ Build Tools:**
    *   Installa i C++ build tools necessari. Puoi farlo scaricando il "Build Tools for Visual Studio" dal [sito di Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio).
    *   Durante l'installazione tramite il Visual Studio Installer, seleziona il carico di lavoro "Sviluppo di applicazioni desktop con C++" e assicurati che i componenti "MSVC..." e "Windows SDK" siano inclusi.

*   **Riavvia il PC**


### 2. Configurazione Ambiente Virtuale

È consigliato creare un ambiente virtuale per isolare le dipendenze del progetto. Assicurati di essere all'interno della cartella dove vuoi eseguire il progetto.

```powershell
# Crea un ambiente virtuale
python -m venv face_recognition_demo
```

```powershell
# Attiva l'ambiente virtuale ogni volta che devi eseguire il codice
# Su Windows
.\face_recognition_demo\Scripts\Activate
```

### 3. Installazione Dipendenze Python

Una volta attivati l'ambiente virtuale e installati i prerequisiti, installa le librerie Python necessarie:

```powershell
# Da eseguire solo la prima volta
pip install face_recognition Pillow chromadb numpy
```

### 4. Lancia la demo

Ora sei pronto per eseguire la demo.
```powershell
.\face_recognition_demo\Scripts\Activate
python main.py
```
