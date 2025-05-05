import face_recognition
import chromadb
import numpy as np
from PIL import Image
import os
import uuid # Per generare ID unici
# Rimosso: from google.colab import files # Non necessario in locale
# Rimosso: from IPython.display import display # Opzionale in locale, sostituito o rimosso

# --- Configurazione Percorsi Locali ---
# Assicurati che queste cartelle esistano nella stessa directory dello script
# o fornisci percorsi assoluti.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory dello script
DATABASE_DIR = os.path.join(SCRIPT_DIR, "database_faces")
QUERY_DIR = os.path.join(SCRIPT_DIR, "query_faces")
DB_PATH = os.path.join(SCRIPT_DIR, "vector_db_local") # Percorso locale per ChromaDB
COLLECTION_NAME = "face_embeddings_local"
SIMILARITY_THRESHOLD = 0.5 # Soglia di distanza (più bassa = più simile). Da aggiustare!

# --- Assicurati che le directory esistano ---
if not os.path.exists(DATABASE_DIR):
    print(f"Cartella '{DATABASE_DIR}' non trovata. Creazione...")
    os.makedirs(DATABASE_DIR)
    print(f"Cartella '{DATABASE_DIR}' creata. Popolala con le immagini del database.")
    # Potresti voler uscire se la cartella è appena stata creata e vuota
    # raise SystemExit("Popola la cartella database e riesegui.")

if not os.path.exists(QUERY_DIR):
    print(f"Cartella '{QUERY_DIR}' non trovata. Creazione...")
    os.makedirs(QUERY_DIR)
    print(f"Cartella '{QUERY_DIR}' creata. Popolala con le immagini di query.")
    # Potresti voler uscire se la cartella è appena stata creata e vuota
    # raise SystemExit("Popola la cartella query e riesegui.")

# --- Funzioni Helper (modificate per display locale opzionale) ---
def show_image_local(image_path, title="Image", size=(150, 150)):
    """Helper per mostrare un'immagine localmente (opzionale)."""
    try:
        img = Image.open(image_path)
        img.thumbnail(size) # Ridimensiona mantenendo l'aspect ratio
        img.show(title=f"{title} - {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Impossibile mostrare l'immagine {image_path}: {e}")

def get_face_embedding(image_path):
    """Carica un'immagine, rileva un volto e restituisce il suo embedding."""
    try:
        print(f"Processing: {image_path}")
        if not os.path.exists(image_path):
             print(f"Errore: File non trovato - {image_path}")
             return None
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"Nessun volto trovato in: {image_path}")
            # show_image_local(image_path, "Nessun Volto Trovato") # Debug opzionale
            return None
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
        if face_encodings:
            # Debug opzionale per vedere il volto rilevato
            # top, right, bottom, left = face_locations[0]
            # face_image = image[top:bottom, left:right]
            # pil_face = Image.fromarray(face_image)
            # pil_face.thumbnail((100,100))
            # pil_face.show(title=f"Volto Rilevato - {os.path.basename(image_path)}")
            return face_encodings[0]
        else:
            print(f"Impossibile calcolare l'embedding per: {image_path}")
            return None
    except FileNotFoundError:
         print(f"Errore critico: File non trovato durante il caricamento - {image_path}")
         return None
    except Exception as e:
        print(f"Errore durante l'elaborazione di {image_path}: {e}")
        return None

# --- Inizializzazione Database ---
print(f"Inizializzazione ChromaDB in: {DB_PATH}")
# Usa PersistentClient per salvare i dati localmente nel percorso specificato
client = chromadb.PersistentClient(path=DB_PATH)

# Ottieni o crea la collezione
try:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"} # Specifica la metrica di distanza
    )
    print(f"Collezione '{COLLECTION_NAME}' ottenuta o creata.")
    # Svuota la collezione se esiste già per rieseguire la demo da zero (opzionale)
    # existing_ids = collection.get()['ids']
    # if existing_ids:
    #     print(f"Svuotamento collezione esistente con {len(existing_ids)} elementi...")
    #     collection.delete(ids=existing_ids)
    #     print("Collezione svuotata.")

except Exception as e:
    print(f"Errore durante l'inizializzazione di ChromaDB: {e}")
    raise SystemExit("Errore critico ChromaDB.")


# --- Popolamento Database ---
print("\nPopolamento del database vettoriale...")
populated_ids = set(collection.get(include=[])['ids']) # Ottieni solo gli ID
added_count = 0
if os.path.exists(DATABASE_DIR):
    image_files = [f for f in os.listdir(DATABASE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Nessun file immagine trovato in '{DATABASE_DIR}'.")
    else:
        for filename in image_files:
            # Usa il nome del file (senza estensione) come ID/nome persona
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(DATABASE_DIR, filename)

            # Controlla se l'ID è già nel DB
            if person_name in populated_ids:
                print(f"'{person_name}' già presente nel database, skipping.")
                continue

            embedding = get_face_embedding(image_path)
            if embedding is not None:
                try:
                    collection.add(
                        embeddings=[embedding.tolist()],
                        metadatas=[{"name": person_name, "source_file": filename}],
                        ids=[person_name] # Usa il nome del file come ID univoco
                    )
                    print(f"Aggiunto '{person_name}' al database.")
                    added_count += 1
                    populated_ids.add(person_name) # Aggiorna il set locale
                except Exception as e:
                     print(f"Errore durante l'aggiunta di {person_name} a ChromaDB: {e}")
else:
    # Questo non dovrebbe accadere grazie al controllo e creazione iniziale
    print(f"ERRORE: La cartella del database '{DATABASE_DIR}' non esiste.")


if added_count > 0:
    print(f"Aggiunti {added_count} nuovi volti al database.")
elif not populated_ids and not image_files: # Controlla se c'erano file da processare
     print("Nessun nuovo volto aggiunto e il database era/è vuoto.")
elif not added_count and populated_ids:
    print("Nessun nuovo volto aggiunto, il database contiene volti da esecuzioni precedenti.")
else: # Nessun nuovo volto aggiunto, ma il DB era vuoto e c'erano file (falliti?)
    print("Nessun nuovo volto aggiunto. Controllare i log per errori di embedding.")


print(f"Elementi totali nel database: {collection.count()}")


# --- Query ---
print("\nEsecuzione query...")
if os.path.exists(QUERY_DIR):
    query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not query_files:
        print(f"Nessun file immagine trovato in '{QUERY_DIR}'.")
    else:
        for filename in query_files:
            query_image_path = os.path.join(QUERY_DIR, filename)
            print(f"\n--- Query con: {filename} ---")

            # Mostra l'immagine di query (opzionale, apre in una finestra separata)
            # show_image_local(query_image_path, "Query Image", size=(150, 150))

            query_embedding = get_face_embedding(query_image_path)

            if query_embedding is not None:
                try:
                    # Cerca i vicini più prossimi
                    results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=1 # Trova solo il vicino più prossimo
                    )

                    # Analizza i risultati
                    if results and results.get('ids') and results['ids'][0]:
                        # ChromaDB restituisce liste di liste, prendiamo il primo risultato [0][0]
                        match_id = results['ids'][0][0]
                        match_distance = results['distances'][0][0]
                        match_metadata = results['metadatas'][0][0]
                        match_name = match_metadata.get('name', 'Nome Sconosciuto')
                        match_source_file = match_metadata.get('source_file', 'File Sconosciuto')

                        print(f"  Volto più simile trovato nel DB: '{match_name}' (da {match_source_file})")
                        print(f"  Distanza: {match_distance:.4f}")

                        # Mostra l'immagine corrispondente dal database (opzionale)
                        match_image_path = os.path.join(DATABASE_DIR, match_source_file)
                        if os.path.exists(match_image_path):
                             print("  Immagine corrispondente nel DB:")
                             # show_image_local(match_image_path, "Match Found", size=(100, 100))
                        else:
                             print(f"  Immagine DB ({match_image_path}) non trovata.")


                        # Confronta la distanza con la soglia
                        if match_distance <= SIMILARITY_THRESHOLD:
                            print(f"  Risultato: CORRISPONDENZA TROVATA - Probabilmente è {match_name}")
                        else:
                            print(f"  Risultato: NESSUNA CORRISPONDENZA (distanza > soglia {SIMILARITY_THRESHOLD})")
                    else:
                        print("  Nessun risultato trovato nel database per questa query.")

                except Exception as e:
                    print(f"Errore durante la query a ChromaDB per {filename}: {e}")
            else:
                print(f"  Impossibile ottenere l'embedding per l'immagine query: {filename}")
else:
     # Questo non dovrebbe accadere grazie al controllo e creazione iniziale
     print(f"ERRORE: La cartella query '{QUERY_DIR}' non esiste.")

print("\nDemo completata.")