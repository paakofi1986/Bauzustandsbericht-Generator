# Bauzustandsbericht Generator – Bildgetriebene Analyse


Eine Streamlit-App, die aus Bildern Bauteile erkennt (nur solche, die in einer Excel-Spalte **„Bauteil“** vorkommen) und pro Bauteil eine normnahe Zusammenfassung nach **„Kategorie“** erstellt.


## Features
- Upload von Excel (Spalten **Bauteil**, **Kategorie**)
- Upload mehrerer Bilder (JPG/PNG/WebP)
- Galerie-Ansicht
- Auswahl Ausgabeformat: Fließtext oder Bulletpoints
- Berichts-Download als TXT


## Tech-Stack
- Python, Streamlit
- OpenAI API (Vision + Text)
- pandas, Pillow, openpyxl


## Voraussetzungen
- Python 3.10+
- OpenAI API Key


## Installation
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
