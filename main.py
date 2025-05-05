import face_recognition
import chromadb
import numpy as np
from PIL import Image
import os
import sys # Aggiunto per sys.exit()

# --- Configurazione Globale ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "database_faces")
QUERY_DIR = os.path.join(SCRIPT_DIR, "query_faces")
DB_PATH = os.path.join(SCRIPT_DIR, "vector_db_local")
COLLECTION_NAME = "face_embeddings_local"
SIMILARITY_THRESHOLD = 0.5

# --- Funzioni Helper (Invariate) ---
def show_image_local(image_path, title="Image", size=(150, 150)):
    """Helper per mostrare un'immagine localmente (opzionale)."""
    try:
        img = Image.open(image_path)
        img.thumbnail(size)
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
            return None
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
        if face_encodings:
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

# --- Funzioni Principali ---

def initialize_chromadb():
    """Inizializza ChromaDB, crea le cartelle se necessario e restituisce client e collezione."""
    print("--- Inizializzazione Ambiente ---")
    # Assicurati che le directory esistano
    for dir_path in [DATABASE_DIR, QUERY_DIR]:
        if not os.path.exists(dir_path):
            print(f"Cartella '{os.path.basename(dir_path)}' non trovata. Creazione...")
            os.makedirs(dir_path)
            print(f"Cartella '{os.path.basename(dir_path)}' creata. Popolala con le immagini appropriate.")

    print(f"\nInizializzazione ChromaDB in: {DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"}
        )
        print(f"Collezione '{COLLECTION_NAME}' ottenuta o creata.")
        print(f"Elementi attuali nella collezione: {collection.count()}")
        print("--- Inizializzazione Completata ---")
        return client, collection
    except Exception as e:
        print(f"Errore critico durante l'inizializzazione di ChromaDB: {e}")
        return None, None

def populate_database(collection):
    """Popola il database vettoriale con le immagini dalla cartella DATABASE_DIR."""
    if not collection:
        print("Errore: Collezione non inizializzata. Esegui prima l'inizializzazione.")
        return

    print("\n--- Popolamento Database ---")
    if not os.path.exists(DATABASE_DIR):
        print(f"ERRORE: La cartella del database '{DATABASE_DIR}' non esiste.")
        return

    populated_ids = set(collection.get(include=[])['ids'])
    added_count = 0
    image_files = [f for f in os.listdir(DATABASE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Nessun file immagine trovato in '{DATABASE_DIR}'.")
    else:
        print(f"Trovati {len(image_files)} file immagine in '{DATABASE_DIR}'.")
        for filename in image_files:
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(DATABASE_DIR, filename)

            if person_name in populated_ids:
                print(f"'{person_name}' già presente nel database, skipping.")
                continue

            embedding = get_face_embedding(image_path)
            if embedding is not None:
                try:
                    collection.add(
                        embeddings=[embedding.tolist()],
                        metadatas=[{"name": person_name, "source_file": filename}],
                        ids=[person_name]
                    )
                    print(f"Aggiunto '{person_name}' al database.")
                    added_count += 1
                    populated_ids.add(person_name)
                except Exception as e:
                     print(f"Errore durante l'aggiunta di {person_name} a ChromaDB: {e}")
            # else: # Messaggio già stampato da get_face_embedding
            #    print(f"Skipping {filename} a causa di errore nell'embedding.")

    if added_count > 0:
        print(f"\nAggiunti {added_count} nuovi volti al database.")
    elif not populated_ids and not image_files:
         print("\nNessun nuovo volto aggiunto e il database era/è vuoto.")
    elif not added_count and populated_ids:
        print("\nNessun nuovo volto aggiunto, il database contiene volti da esecuzioni precedenti.")
    else:
        print("\nNessun nuovo volto aggiunto. Controllare i log per errori di embedding.")

    print(f"Elementi totali nel database: {collection.count()}")
    print("--- Popolamento Completato ---")


def query_images(collection):
    """Esegue query con le immagini dalla cartella QUERY_DIR."""
    if not collection:
        print("Errore: Collezione non inizializzata. Esegui prima l'inizializzazione.")
        return
    if collection.count() == 0:
        print("Attenzione: La collezione è vuota. Popola prima il database.")
        # Potresti voler continuare comunque se ha senso nel tuo caso d'uso
        # return

    print("\n--- Esecuzione Query ---")
    if not os.path.exists(QUERY_DIR):
        print(f"ERRORE: La cartella query '{QUERY_DIR}' non esiste.")
        return

    query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not query_files:
        print(f"Nessun file immagine trovato in '{QUERY_DIR}'.")
    else:
        print(f"Trovati {len(query_files)} file immagine di query in '{QUERY_DIR}'.")
        for filename in query_files:
            query_image_path = os.path.join(QUERY_DIR, filename)
            print(f"\n--- Query con: {filename} ---")

            # Opzionale: Mostra immagine query
            # show_image_local(query_image_path, "Query Image", size=(150, 150))

            query_embedding = get_face_embedding(query_image_path)

            if query_embedding is not None:
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=1
                    )

                    if results and results.get('ids') and results['ids'][0]:
                        match_id = results['ids'][0][0]
                        match_distance = results['distances'][0][0]
                        match_metadata = results['metadatas'][0][0]
                        match_name = match_metadata.get('name', 'Nome Sconosciuto')
                        match_source_file = match_metadata.get('source_file', 'File Sconosciuto')

                        print(f"  Volto più simile trovato nel DB: '{match_name}' (da {match_source_file})")
                        print(f"  Distanza: {match_distance:.4f}")

                        # Opzionale: Mostra immagine match
                        # match_image_path = os.path.join(DATABASE_DIR, match_source_file)
                        # if os.path.exists(match_image_path):
                        #     show_image_local(match_image_path, "Match Found", size=(100, 100))

                        if match_distance <= SIMILARITY_THRESHOLD:
                            print(f"  Risultato: CORRISPONDENZA TROVATA (distanza <= {SIMILARITY_THRESHOLD})")
                        else:
                            print(f"  Risultato: NESSUNA CORRISPONDENZA (distanza > {SIMILARITY_THRESHOLD})")
                    else:
                        print("  Nessun risultato trovato nel database per questa query.")

                except Exception as e:
                    print(f"Errore durante la query a ChromaDB per {filename}: {e}")
            # else: # Messaggio già stampato da get_face_embedding
            #    print(f"  Impossibile ottenere l'embedding per l'immagine query: {filename}")

    print("--- Query Completate ---")

# --- Menu Principale ---
def main_menu():
    """Mostra il menu e gestisce l'input dell'utente."""
    client = None
    collection = None

    while True:
        print("\n--- MENU DEMO FACE RECOGNITION ---")
        print("1. Inizializza Ambiente e ChromaDB")
        print("2. Popola Database (richiede inizializzazione)")
        print("3. Esegui Query (richiede inizializzazione)")
        print("4. Esci")

        scelta = input("Scegli un'opzione: ")

        if scelta == '1':
            client, collection = initialize_chromadb()
        elif scelta == '2':
            if collection:
                populate_database(collection)
            else:
                print("Errore: Devi prima inizializzare l'ambiente (opzione 1).")
        elif scelta == '3':
            if collection:
                query_images(collection)
            else:
                print("Errore: Devi prima inizializzare l'ambiente (opzione 1).")
        elif scelta == '4':
            print("Uscita dalla demo.")
            sys.exit() # Esce dallo script
        else:
            print("Scelta non valida. Riprova.")

# --- Esecuzione ---
if __name__ == "__main__":
    main_menu()