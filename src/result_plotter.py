import json
import matplotlib.pyplot as plt
import numpy as np


# Funktion zum Einlesen des JSON-Files
def load_json_from_file(file_path):
    """
    Liest das JSON aus einer Datei und gibt es als Python-Datenstruktur zurück.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Funktion zum Erstellen des Plots für 'correct'/'false' und 'average_distance_non_correct'
def plot_element_data(data, element_name, ax1, ax2):
    """
    Erstellt zwei Bar-Charts:
    1. Für 'correct' und 'false' für das angegebene Element.
    2. Für 'average_distance_non_correct' für das angegebene Element.
    """
    # Extrahiere alle Kombinationen von length_scale, n_rotations und n_scale
    combinations = []
    correct_values = []
    false_values = []
    avg_distance_non_correct_values = []

    for item in data:
        length_scale = item["length_scale"]
        n_rotations = item["n_rotations"]
        n_scale = item["n_scale"]

        # Hole die Werte für das angegebene Element (z.B. 'agent', 'walls', etc.)
        element_data = item["Elements"].get(element_name)
        if element_data:
            correct = element_data["correct"]
            false = element_data["false"]
            avg_distance_non_correct = element_data["average_distance_non_correct"]

            # Speichere die Kombinationen und die zugehörigen Werte
            combinations.append(f"LS={length_scale}, NR={n_rotations}, NS={n_scale}")
            correct_values.append(correct)
            false_values.append(false)
            avg_distance_non_correct_values.append(avg_distance_non_correct)
        else:
            print(f"Element '{element_name}' nicht in den Daten gefunden!")

    # Erstelle einen Index für die x-Achse (Positionen der Balken)
    x = np.arange(len(combinations))  # Position der Balken
    width = 0.35  # Breite der Balken

    # Erstes Bar-Chart (correct vs. false)
    ax1.bar(x - width / 2, correct_values, width, label='Correct', color='g')
    ax1.bar(x + width / 2, false_values, width, label='False', color='r')

    # Hinzufügen von Text für die Werte
    ax1.bar_label(ax1.containers[0])
    ax1.bar_label(ax1.containers[1])

    # Achsenbezeichner und Titel für den ersten Plot
    ax1.set_ylabel('Anzahl')
    ax1.set_title(
        f'{element_name.capitalize()} - Correct vs False für verschiedene Kombinationen von length_scale, n_rotations und n_scale')
    ax1.set_xticks(x)
    ax1.set_xticklabels(combinations, rotation=45, ha="right")
    ax1.legend()

    # Zweites Bar-Chart (average_distance_non_correct)
    ax2.bar(x, avg_distance_non_correct_values, width, label='Average Distance Non-Correct', color='b')

    # Hinzufügen von Text für die Werte
    ax2.bar_label(ax2.containers[0])

    # Achsenbezeichner und Titel für den zweiten Plot
    ax2.set_ylabel('Durchschnittliche Entfernung (Nicht korrekt)')
    ax2.set_title(
        f'{element_name.capitalize()} - Average Distance Non-Correct für verschiedene Kombinationen von length_scale, n_rotations und n_scale')
    ax2.set_xticks(x)
    ax2.set_xticklabels(combinations, rotation=45, ha="right")


# Funktion zum Erstellen der Subplots für mehrere Elemente
def plot_multiple_elements(data, element_names):
    """
    Erstellt Subplots für mehrere Elemente, die untereinander angezeigt werden, mit je einem zusätzlichen Plot rechts.
    """
    # Anzahl der Subplots (für jedes Element ein Subplot-Paar)
    num_elements = len(element_names)

    # Erstellen der Subplots (num_elements Zeilen, 2 Spalten)
    fig, axes = plt.subplots(num_elements, 2, figsize=(16, 6 * num_elements))

    # Falls nur ein Element, wird axes ein Array mit einem Element sein, daher in eine Liste umwandeln
    if num_elements == 1:
        axes = [axes]

    # Iteriere über alle Elemente und erstelle für jedes Element einen Plot
    for i, element_name in enumerate(element_names):
        plot_element_data(data, element_name, axes[i][0], axes[i][1])

    # Automatische Anpassung der Layouts
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.show()


# Beispielaufruf: JSON-Datei laden und Plots für mehrere Elemente erstellen
data = load_json_from_file("output_images/results.json")

# Liste von Elementen, die geplottet werden sollen
element_names = ['walls', 'agent']  # Füge hier weitere Elemente hinzu, falls gewünscht
plot_multiple_elements(data, element_names)


