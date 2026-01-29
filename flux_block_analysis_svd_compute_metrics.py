import os
import json
from pathlib import Path
from tqdm import tqdm  # Optional: pip install tqdm

# --- 1. Deine Berechnungs-Funktionen (Platzhalter) ---

def calculate_cmmd(folder_path, reference_path=None):
    # Ersetze dies durch deinen echten Aufruf
    return 0.45  # Dummy-Wert

def calculate_clip(folder_path, reference_path=None):
    # Ersetze dies durch deinen echten Aufruf
    return 30.1  # Dummy-Wert

# --- 2. Hauptlogik ---

def compute_metrics_separate(main_folder_path, reference_folder_path=None):
    """
    Erstellt zwei separate Dictionaries für CMMD und CLIP.
    Key ist jeweils der Name des Unterordners.
    """
    main_path = Path(main_folder_path)
    
    if not main_path.exists():
        raise FileNotFoundError(f"Der Pfad {main_folder_path} existiert nicht.")

    # Nur echte Unterordner filtern
    subfolders = [p for p in main_path.iterdir() if p.is_dir()]
    
    # Initialisierung der getrennten Dictionaries
    cmmd_dict = {}
    clip_dict = {}

    print(f"Verarbeite {len(subfolders)} Unterordner in '{main_path.name}'...\n")

    for subfolder in tqdm(subfolders, desc="Berechne Metriken"):
        folder_name = subfolder.name
        
        try:
            # 1. CMMD berechnen und in das CMMD-Dict speichern
            cmmd_val = calculate_cmmd(subfolder, reference_folder_path)
            cmmd_dict[folder_name] = cmmd_val
            
            # 2. CLIP berechnen und in das CLIP-Dict speichern
            clip_val = calculate_clip(subfolder, reference_folder_path)
            clip_dict[folder_name] = clip_val
            
        except Exception as e:
            print(f"Fehler bei '{folder_name}': {e}")
            # Optional: Fehlermeldung als Wert speichern, damit der Key existiert
            cmmd_dict[folder_name] = None
            clip_dict[folder_name] = None

    return cmmd_dict, clip_dict

# --- 3. Ausführung ---

if __name__ == "__main__":
    # Pfade anpassen
    root_directory = "./experimente"
    ref_directory = "./referenz_bilder"
    
    # Funktionsaufruf: Gibt zwei Dicts zurück
    results_cmmd, results_clip = compute_metrics_separate(root_directory, ref_directory)
    
    # --- Ausgabe zur Überprüfung ---
    
    print("\n--- CMMD Ergebnisse ---")
    print(json.dumps(results_cmmd, indent=4))
    
    print("\n--- CLIP Ergebnisse ---")
    print(json.dumps(results_clip, indent=4))