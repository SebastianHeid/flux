import os
import argparse
import re
import sys

def rename_files_generic(directory, dry_run=False):
    if not os.path.exists(directory):
        print(f"Fehler: Der Ordner '{directory}' existiert nicht.")
        sys.exit(1)

    print(f"Scanne Ordner: {directory}")
    print("Modus: Suche nach '..._ZAHL.png' -> 'prompt_ZAHL.png'")
    print("-" * 50)

    # Regex: Findet eine Zahl (\d+) direkt vor dem .png am Ende der Zeile ($)
    # Egal was davor steht.
    pattern = re.compile(r".*?(\d+)\.png$")
    
    count = 0
    renamed_count = 0
    
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".png")])

    for filename in files:
        match = pattern.search(filename)
        
        if match:
            # Die gefundene Zahl extrahieren (z.B. "5" aus "bild_5.png")
            number_i = match.group(1)
            
            # Ziel-Name
            new_filename = f"prompt_{number_i}.png"
            
            # Verhindern, dass wir Dateien umbenennen, die schon richtig heißen
            if filename == new_filename:
                continue

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            if dry_run:
                print(f"[Dry Run] '{filename}'  --->  '{new_filename}'")
            else:
                try:
                    os.rename(old_path, new_path)
                    # Optional: Kleiner Print alle 100 Dateien, um Spam zu vermeiden
                    # print(f"[OK] '{filename}' -> '{new_filename}'") 
                    renamed_count += 1
                except OSError as e:
                    print(f"[Fehler] Konnte '{filename}' nicht umbenennen: {e}")
            
            count += 1
        else:
            print(f"[Skip] Keine Zahl am Ende gefunden: '{filename}'")

    print("-" * 50)
    if dry_run:
        print(f"Testlauf beendet. {count} Dateien würden umbenannt werden.")
    else:
        print(f"Fertig. {renamed_count} Dateien erfolgreich umbenannt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benennt alle PNGs mit einer End-Zahl zu 'prompt_i.png' um.")
    
    parser.add_argument("folder", type=str, help="Pfad zum Ordner mit den Bildern")
    parser.add_argument("--dry-run", action="store_true", help="Zeigt nur an, was passieren würde")
    
    args = parser.parse_args()
    
    rename_files_generic(args.folder, args.dry_run)