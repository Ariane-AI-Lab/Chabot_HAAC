import google.generativeai as genai
import os
from dotenv import load_dotenv

# Charge le .env qui se trouve dans le même dossier que ce script
load_dotenv() 

api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    print("❌ Erreur : GENAI_API_KEY non trouvée dans le fichier .env")
else:
    print(f"✅ Clé trouvée (début) : {api_key[:10]}...")
    genai.configure(api_key=api_key)
    
    try:
        print("\n--- Modèles de texte disponibles ---")
        for m in genai.list_models():
            # On cherche les modèles capables de générer du contenu (comme flash ou pro)
            if 'generateContent' in m.supported_generation_methods:
                print(f"Modèle : {m.name}")
                
        print("\n--- Modèles d'embeddings disponibles ---")
        for m in genai.list_models():
            if 'embed' in m.name:
                print(f"Embedding : {m.name}")
    except Exception as e:
        print(f"Erreur API : {e}")