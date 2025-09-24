# app.py
import os
import io
import json
import base64
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------
# Page / Header
# ---------------------------------
st.set_page_config(page_title="Bauzustandsbericht Generator (Bildgetrieben)", layout="wide")
st.title("🏗️ Bauzustandsbericht Generator – Bildgetriebene Analyse")
st.caption("Analysiert NUR die hochgeladenen Bilder. Erkannt werden ausschließlich Bauteile aus der Excel-Spalte „Bauteil“. Zusammenfassung je Bauteil nach „Kategorie“.")

# ---------------------------------
# API Key
# ---------------------------------
env_key = os.getenv("OPENAI_API_KEY", "")
api_key = st.text_input("🔑 OpenAI API Key (leer lassen, wenn per Umgebung gesetzt)", value=env_key, type="password")
client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Fehler beim Initialisieren des OpenAI Clients: {e}")

# ---------------------------------
# Uploads
# ---------------------------------
uploaded_excel = st.file_uploader("📂 Excel (Spalten: 'Bauteil', 'Kategorie')", type=["xlsx"])
images = st.file_uploader("📸 Bilder (mehrfach möglich)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

# ---------------------------------
# Ausgabeformat
# ---------------------------------
mode = st.radio("📑 Ausgabeformat", ["Fliesstext", "Bulletpoints"], horizontal=True)

# ---------------------------------
# Utils
# ---------------------------------
def thumb_b64(uploaded_file, max_size=(520, 520)) -> str:
    """
    Erzeugt ein Base64-Thumbnail aus dem hochgeladenen Bild (JPEG).
    Robust gegen Re-Runs (nutzt getvalue()).
    """
    data = uploaded_file.getvalue() or uploaded_file.read()
    if not data:
        return ""
    try:
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(data)).convert("RGB")
        img.thumbnail(max_size)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Fallback: Originalbytes
        return base64.b64encode(data).decode("utf-8")

def to_b64(uploaded_file) -> Tuple[str, str]:
    """
    Liefert (mime, base64) des Originalbilds.
    """
    data = uploaded_file.getvalue() or uploaded_file.read()
    if not data:
        return "image/jpeg", ""
    mime = uploaded_file.type or "image/jpeg"
    return mime, base64.b64encode(data).decode("utf-8")

def show_one_row_gallery(images):
    st.markdown("### 🖼️ Bilder-Galerie")
    st.markdown("""
    <style>
      .hscroll-wrap {
          display: block; overflow-x: auto; overflow-y: hidden; white-space: nowrap;
          padding: 6px 2px 8px 2px; border-bottom: 1px solid #e6e6e6;
      }
      .tile { display: inline-block; vertical-align: top; margin: 0 8px 0 0; text-align: center; }
      .tile img {
          height: 160px; border-radius: 10px; object-fit: cover;
          box-shadow: 0 1px 4px rgba(0,0,0,0.15); display: block;
      }
      .cap {
          font-size: 12px; color: #666; margin-top: 4px; max-width: 240px;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }
    </style>
    """, unsafe_allow_html=True)

    thumbs_b64 = [(img.name, thumb_b64(img)) for img in images]
    html = ['<div class="hscroll-wrap">']
    for name, b64 in thumbs_b64:
        if not b64:
            continue
        html.append(f'<div class="tile"><img src="data:image/jpeg;base64,{b64}" alt="{name}"/><div class="cap">{name}</div></div>')
    html.append('</div>')
    st.markdown("\n".join(html), unsafe_allow_html=True)

def build_catalog(df: pd.DataFrame) -> Dict[str, str]:
    """
    Baut ein Lexikon: {Bauteil: Kategorieanforderungen}
    Nur exakt die in Excel vorhandenen Bauteile.
    """
    catalog = {}
    for _, r in df.iterrows():
        b = str(r["Bauteil"]).strip()
        k = str(r["Kategorie"]).strip()
        if b:
            catalog[b] = k
    return catalog

def build_detection_prompt(bauteile: List[str]) -> str:
    """
    Kurzer, strenger Prompt für die Bild-Erkennung.
    Output zwingend als JSON liefern.
    """
    joined = ", ".join(sorted(set(bauteile)))
    return f"""
Du bist Bau-/Immobiliengutachter (Vision). Erkenne auf DEM EINZELNEN Bild ausschließlich Bauteile aus dieser Liste:
{joined}

Gib ein JSON mit folgendem Format zurück:
{{
  "detected": [
    {{
      "bauteil": "<eines aus der Liste exakt>",
      "evidence": "kurze visuelle Hinweise (2-4 Bullet-ähnliche Phrasen)",
      "zustand": "knapp: intakt / gebraucht / schadhaft / unklar (optional kurz warum)"
    }}
  ]
}}

WICHTIG:
- Wenn kein gelisteter Bauteil sichtbar ist, gib {{ "detected": [] }} zurück.
- Keine Erfindungen. Nur Bauteile aus der Liste zulassen.
- JSON strikt ohne Zusatztext.
""".strip()

def parse_json_loose(txt: str) -> Dict[str, Any]:
    """
    Versucht tolerant JSON aus einem Modell-Output zu extrahieren.
    """
    # direkter Versuch
    try:
        return json.loads(txt)
    except Exception:
        pass
    # innerhalb von Code-Blöcken
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"detected": []}

def analyze_single_image(client: OpenAI, model: str, img_b64: str, mime: str, bauteile: List[str]) -> List[Dict[str, Any]]:
    """
    Ruft die Chat-Completion mit Vision an: ein Bild + knapper Erkennungs-Prompt.
    """
    if not img_b64:
        return []
    prompt = build_detection_prompt(bauteile)
    try:
        resp = client.chat.completions.create(
            model=model,  # z.B. "gpt-5"
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein präziser, normnaher Bau-/Immobiliengutachter (Vision). Antworte, wo gefordert, NUR im JSON-Format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }
            ],
        )
        raw = resp.choices[0].message.content.strip()
        data = parse_json_loose(raw)
        return data.get("detected", [])
    except Exception as e:
        # Für Transparenz im UI protokollieren
        st.warning(f"Vision-Analyse für ein Bild fehlgeschlagen: {e}")
        return []

def build_synthesis_prompt(catalog: Dict[str, str], evidence_map: Dict[str, List[Dict[str, str]]], mode: str) -> str:
    """
    Baut den Prompt für die abschließende Synthese:
    - Nur Bauteile, für die es Evidenz (aus Bildern) gibt.
    - Nutzung der Kategorie-Vorgaben pro Bauteil.
    - Ausgabeformat wählbar (Fliesstext oder Bulletpoints).
    """
    lines = []
    for bauteil, hits in evidence_map.items():
        if not hits:
            continue
        lines.append(f"### {bauteil}")
        lines.append(f"Kategorie-Vorgaben: {catalog.get(bauteil, '')}")
        lines.append("Bildbefunde (kondensiert):")
        for h in hits:
            ev = h.get("evidence", "")
            zs = h.get("zustand", "")
            lines.append(f"- Evidenz: {ev} | Zustand: {zs}")
        lines.append("")

    style = (
        "Schreibe die abschließende Darstellung je Bauteil als zusammenhängenden Fliesstext, normnah, wie in Bauzustandsberichten."
        if mode == "Fliesstext"
        else "Schreibe die abschließende Darstellung je Bauteil als klare Bulletpoints (knapp, technisch, normnah)."
    )

    return f"""
Du bist ein sachlicher, technisch präziser Bau-/Immobiliengutachter.
Unten siehst du pro Bauteil die zusammengefassten Bildbefunde (nur Bauteile, die auf mind. einem Bild erkannt wurden) 
sowie die jeweiligen Kategorie-Vorgaben aus Excel.

DEINE AUFGABE:
- Erstelle je Bauteil einen **Bauteilbericht**, der sich strikt an die Kategorie-Vorgaben hält.
- Fasse die Evidenzen aus allen Bildern zusammen; widersprüchliches benenne vorsichtig.
- Keine Bauteile erfinden. Nur jene Bauteile behandeln, die unten aufgeführt sind.
- Wenn eine Angabe unsicher ist, nutze Formulierungen wie „vermutlich“ oder „geschätzt“.
- Nutze ggf. normative Begriffe (z. B. elektrische Sicherheit, Wartungspflichten, Hygienerisiken), bleibe aber sachlich.

{style}

MATERIAL:
{os.linesep.join(lines)}
""".strip()

st.divider()

# ---------------------------------
# Galerie
# ---------------------------------
if images:
    show_one_row_gallery(images)
else:
    st.info("Bitte zuerst Bilder hochladen – sie erscheinen dann hier als horizontale Galerie.")


# ---------------------------------
# Excel-Vorschau & Katalog
# ---------------------------------
df = None
catalog: Dict[str, str] = {}
if uploaded_excel:
    try:
        df = pd.read_excel(uploaded_excel)
        if not {"Bauteil", "Kategorie"}.issubset(df.columns):
            st.error("❌ In der Excel fehlen die Spalten **'Bauteil'** und/oder **'Kategorie'**.")
            df = None
        else:
            st.markdown("### Ausgelesene Bauteile & Kategorien")
            st.dataframe(df[["Bauteil", "Kategorie"]], use_container_width=True)
            catalog = build_catalog(df)
    except ImportError:
        st.error("Fehlende Excel-Engine (openpyxl). Bitte installieren: `pip install openpyxl`")
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Excel-Datei: {e}")
    st.divider()
# ---------------------------------
# Analyse starten
# ---------------------------------
start = st.button("🚀 Bildanalyse & Bericht erstellen", type="primary", disabled=client is None or df is None or not images)
if start:
    if client is None:
        st.error("Bitte zuerst einen gültigen OpenAI API Key eingeben.")
    elif df is None or not catalog:
        st.error("Bitte zuerst eine Excel-Datei mit 'Bauteil' & 'Kategorie' hochladen.")
    elif not images:
        st.error("Bitte zuerst Bilder hochladen.")
    else:
        with st.spinner("👀 Analysiere Bilder …"):
            # 1) Pro Bild: Vision-Analyse → erkannte Bauteile (nur aus Katalog)
            evidence_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            bauteil_list = list(catalog.keys())

            for img in images:
                mime, b64 = to_b64(img)
                detected = analyze_single_image(client, "gpt-5", b64, mime, bauteil_list)
                for d in detected:
                    bt = d.get("bauteil", "")
                    if bt in catalog:
                        evidence_map[bt].append({
                            "image": img.name,
                            "evidence": d.get("evidence", ""),
                            "zustand": d.get("zustand", "")
                        })

            # Prüfen, ob überhaupt etwas gefunden wurde
            if not any(evidence_map.values()):
                st.warning("Es wurden in den Bildern keine Bauteile aus der Excel-Liste erkannt.")
            else:
                # 2) Synthese: Gesamtbericht NUR für erkannte Bauteile
                synth_prompt = build_synthesis_prompt(catalog, evidence_map, mode)
                try:
                    resp = client.chat.completions.create(
                        model="gpt-5",  # kein temperature-Param: gpt-5 nutzt Default
                        messages=[
                            {"role": "system", "content": "Du bist ein erfahrener, normnaher Bau-/Immobiliengutachter."},
                            {"role": "user", "content": synth_prompt},
                        ],
                    )
                    report = resp.choices[0].message.content
                    st.subheader("📄 Ergebnis – Bauteilberichte (nur erkannte Bauteile)")
                    st.write(report)
                    st.download_button("📥 Bericht als TXT herunterladen", report, file_name="Bauzustandsbericht_bildgetrieben.txt")

                    # Optionale Transparenz: Roh-Evidenzen anzeigen
                    with st.expander("🔎 Detektions-Evidenzen je Bauteil (aus allen Bildern)"):
                        for bt, hits in evidence_map.items():
                            st.markdown(f"**{bt}**")
                            for h in hits:
                                st.markdown(f"- *{h['image']}*: Evidenz: {h['evidence']} | Zustand: {h['zustand']}")
                except Exception as e:
                    st.error(f"Fehler bei der OpenAI-Synthese: {e}")
