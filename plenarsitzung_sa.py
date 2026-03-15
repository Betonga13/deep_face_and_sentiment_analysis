import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. XML-Datei einlesen
xml_file = "21057.xml"  # Pfad zur heruntergeladenen Datei
tree = ET.parse(xml_file)
root = tree.getroot()

# 2. Alle relevanten Texte sammeln
all_text = []

# Beispiel: Angenommen, die Reden stehen in Elementen namens "rede" oder "text"
for elem in root.iter():
    if elem.text:
        all_text.append(elem.text.strip())

full_text = " ".join(all_text)

# 3. Text bereinigen (Sonderzeichen + Kleinbuchstaben)
clean_text = re.sub(r"[^A-Za-zÄäÖöÜüß\s]+", " ", full_text)
clean_text = clean_text.lower()

# 4. Tokenisierung
tokens = word_tokenize(clean_text)

# 5. Stoppwörter entfernen
german_stopwords = set(stopwords.words('german'))
filtered_tokens = [token for token in tokens if token not in german_stopwords and len(token) > 1]

# 6. Text für Analyse wieder zusammensetzen
processed_text = " ".join(filtered_tokens)

# 7. Sentiment mit VADER analysieren
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(processed_text)

print("Tokenisierte Wörter:", filtered_tokens[:50])
print("\nSentiment Scores:", scores)
