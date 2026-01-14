import pandas as pd
import re
import os

def generate_tunisie_pro():
    """Génère le dataset tunisie_pro.csv à partir de real_fake_news_tunisie.csv"""

    input_path = "data/real_fake_news_tunisie.csv"
    output_path = "data/tunisie_pro.csv"

    if not os.path.exists(input_path):
        print(f"Fichier source manquant: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Dataset source charge: {len(df)} lignes")

    def clean_text(text):
        """Nettoie le texte en supprimant les elements repetitifs"""
        text = str(text)

        # Supprimer au debut
        text = re.sub(r'^\s*', '', text)

        # Supprimer " ! Partagez immediatement ! " suivi de chiffres a la fin
        text = re.sub(r'\s*! Partagez immediatement ! \d+$', '', text)

        # Nettoyer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # Appliquer le nettoyage
    df['text'] = df['text'].apply(clean_text)

    # Supprimer les doublons
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"Doublons supprimes: {before - len(df)}")

    # Equilibrer les classes si necessaire (mais garder tout pour l'entrainement)
    print("Distribution des labels:")
    print(df['label'].value_counts())

    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset PRO genere: {output_path} ({len(df)} lignes)")

if __name__ == "__main__":
    generate_tunisie_pro()