'''import cv2
from deepface import DeepFace

# Video-Dateiname (muss genau so heißen wie deine Datei!)
video_datei = "bundestag.mp4.mp4"

print(f"Analysiere: {video_datei}")

# Video öffnen
video = cv2.VideoCapture(video_datei)

if not video.isOpened():
    print(f"FEHLER: Video '{video_datei}' nicht gefunden!")
    print("Stelle sicher, dass die Datei im gleichen Ordner ist wie dieses Skript.")
    exit()

print("Video erfolgreich geladen!")
print("Starte Analyse...")

# Hier kommt der Rest des Codes (später)
video.release()'''

import cv2
from deepface import DeepFace
from collections import Counter
import os

# TensorFlow-Warnungen unterdrücken (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Video-Dateiname
video_datei = "bundestag.mp4.mp4"

print(f"Analysiere: {video_datei}")
print("="*50)

# Video öffnen
video = cv2.VideoCapture(video_datei)

if not video.isOpened():
    print(f"FEHLER: Video '{video_datei}' nicht gefunden!")
    print("Stelle sicher, dass die Datei im gleichen Ordner ist wie dieses Skript.")
    exit()

print("Video erfolgreich geladen!")
print("Starte Analyse (das kann einige Minuten dauern)...")
print("="*50)

# Gesichtserkennung laden
gesicht_erkennung = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Gesichtserkennung mit direktem Pfad
gesicht_erkennung = cv2.CascadeClassifier(
    'C:\\video_analyse\\haarcascade_frontalface_default.xml'
)

# Prüfen ob geladen
if gesicht_erkennung.empty():
    print("FEHLER: Gesichtserkennung konnte nicht geladen werden!")
    print("Lade die Datei herunter von:")
    print("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
    exit()

# Ergebnisse speichern
alle_emotionen = []
frame_nummer = 0
analysierte_frames = 0

while True:
    # Nächstes Frame lesen
    erfolg, bild = video.read()
    
    # Wenn kein Frame mehr kommt, Videoende
    if not erfolg:
        break
    
    # Nur jedes 30. Frame analysieren (für Geschwindigkeit)
    if frame_nummer % 30 == 0:
        # In Graustufen umwandeln (für Gesichtserkennung)
        grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
        
        # Gesichter im Bild finden
        gesichter = gesicht_erkennung.detectMultiScale(
            grau, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(60, 60)  # Mindestgröße für Gesichter
        )
        
        # Jedes gefundene Gesicht analysieren
        for (x, y, w, h) in gesichter:
            try:
                # Gesicht aus dem Bild ausschneiden
                gesicht = bild[y:y+h, x:x+w]
                
                # Emotion analysieren
                ergebnis = DeepFace.analyze(
                    gesicht, 
                    actions=['emotion'],
                    enforce_detection=False,  # Kein Fehler wenn unscharf
                    silent=True  # Weniger Ausgaben
                )
                
                # Emotion aus dem Ergebnis extrahieren
                emotion = ergebnis[0]['dominant_emotion']
                alle_emotionen.append(emotion)
                
                # Fortschritt anzeigen (alle 10 Funde)
                if len(alle_emotionen) % 10 == 0:
                    print(f"Frame {frame_nummer}: {emotion} (insgesamt {len(alle_emotionen)} Gesichter)")
                    
            except Exception as e:
                # Fehler ignorieren (z.B. wenn Gesicht zu klein oder unscharf)
                continue
        
        analysierte_frames += 1
        
        # Fortschritt alle 100 Frames
        if analysierte_frames % 100 == 0:
            print(f"▶️ Fortschritt: {analysierte_frames} Frames analysiert, {len(alle_emotionen)} Gesichter gefunden")
    
    frame_nummer += 1
    
    # Optional: Alle 1000 Frames eine kleine Pause
    if frame_nummer % 1000 == 0:
        print(f"  Verarbeite Frame {frame_nummer}...")

# Video freigeben
video.release()

print("\n" + "="*50)
print("ANALYSE ABGESCHLOSSEN")
print("="*50)
print(f"Verarbeitete Frames: {frame_nummer}")
print(f"Analysierte Frames: {analysierte_frames}")
print(f"Gefundene Gesichter: {len(alle_emotionen)}")

# ERGEBNISSE AUSWERTEN
print("\n" + "="*50)
print("ERGEBNISSE")
print("="*50)

if alle_emotionen:
    # Emotionen zählen
    zaehlung = Counter(alle_emotionen)
    gesamt = len(alle_emotionen)
    
    print(f"\n📊 Emotions-Verteilung:")
    print("-" * 30)
    
    # Alle Emotionen mit Prozent anzeigen
    for emotion, anzahl in zaehlung.most_common():
        prozent = (anzahl / gesamt) * 100
        # Balken für visuelle Darstellung
        balken = "█" * int(prozent / 5)
        print(f"{emotion:8} : {anzahl:3} ({prozent:4.1f}%) {balken}")
    
    # Häufigste Emotion
    haeufigste = zaehlung.most_common(1)[0]
    print(f"\n👉 HÄUFIGSTE EMOTION: {haeufigste[0].upper()}")
    
    # Interpretation
    print("\n📝 INTERPRETATION:")
    if haeufigste[0] == 'happy':
        print("😊 Die Stimmung war überwiegend positiv!")
    elif haeufigste[0] in ['angry', 'sad', 'fear']:
        print("😠 Die Stimmung war überwiegend negativ!")
    else:
        print("😐 Die Stimmung war überwiegend neutral.")
    
    # Ergebnisse speichern
    print("\n💾 Speichere Ergebnisse...")
    with open('ergebnisse.txt', 'w', encoding='utf-8') as f:
        f.write("EMOTIONS-ANALYSE BUNDESTAGSVIDEO\n")
        f.write("="*40 + "\n\n")
        for emotion, anzahl in zaehlung.most_common():
            prozent = (anzahl / gesamt) * 100
            f.write(f"{emotion}: {anzahl} ({prozent:.1f}%)\n")
        f.write(f"\nHäufigste Emotion: {haeufigste[0]}")
    
    print("✅ Ergebnisse gespeichert in 'ergebnisse.txt'")
    
else:
    print("❌ Keine Emotionen erkannt!")
    print("\nMögliche Gründe:")
    print("  - Video ist zu dunkel oder unscharf")
    print("  - Keine Gesichter im Video")
    print("  - Andere Video-Qualität probieren (höhere Auflösung)")

print("\n✅ Analyse fertig!")