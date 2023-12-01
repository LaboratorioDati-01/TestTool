import streamlit as st
import pandas as pd
import random

# Funzione principale per eseguire il calcolo
def calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal):
    # (Inserisci qui il codice che hai fornito, adattandolo per utilizzare queste variabili)
    # Definizione delle variabili iniziali relative alla gestione dell'energia.
    ToT_Demand = 4700  # Domanda totale di energia.
    Wind = 1700  # Produzione di energia eolica.
    
    # Parametri per l'importazione di energia.
    Import = 3000  # Quantità di energia importata.
    Max_Import = 3000  # Massima quantità di energia che può essere importata.
    Min_Import = -1000  # Minima quantità di energia che può essere importata (negativo indica esportazione).
    
    # Parametri per la produzione di energia termica.
    Thermal = 0  # Produzione attuale di energia termica.
    Max_Thermal = 800  # Massima produzione di energia termica possibile.
    Min_Thermal = 0  # Minima produzione di energia termica (non può essere negativa).
    
    # Parametri di flessibilità della domanda.
    flex = 0.1  # Fattore di flessibilità, tra 0.1 e 1.
    Band = 0  # Intervallo di flessibilità, compreso tra 10 e 50.
    Slope = 50  # Pendenza, utilizzata per calcolare l'adattamento della domanda.
    
    # Calcolo della domanda flessibile.
    C1 = ToT_Demand * (1 - flex)  # Domanda non flessibile.
    C2 = ToT_Demand * flex  # Domanda flessibile.
    Min_Flex_Demand = -C2 * 0.5  # Minima domanda flessibile.
    Max_Flex_Demand = C2 * 0.2  # Massima domanda flessibile.
    
    # Prezzi dell'energia.
    Price_Import = 40  # Prezzo dell'energia importata.
    Price_Wind = 20  # Prezzo dell'energia eolica.
    Price_Thermal = 120  # Prezzo dell'energia termica.
    
    # Funzioni per generare variazioni percentuali casuali.
    def genera_variazione_demand():
        return random.uniform(-0.4, 0.4)
    
    def genera_variazione_wind():
        return random.uniform(-0.535, 0.62)
    
    # Creazione di vettori di variazioni casuali per domanda e produzione eolica.
    num_variazioni = 1000
    Variazione_demand = [genera_variazione_demand() for _ in range(num_variazioni)]
    Variazione_wind = [genera_variazione_wind() for _ in range(num_variazioni)]
    
    # Calcolo delle variazioni della domanda e della produzione eolica.
    C1e = [C1 * (1 + var) for var in Variazione_demand]
    We = [Wind * (1 + var) for var in Variazione_wind]
    
    # Calcolo dell'importazione di energia considerando le variazioni.
    Ie = [max(min(C1e[k] + C2 - We[k], Max_Import), Min_Import) for k in range(1000)]
    
    # Calcolo di un parametro relativo alla produzione totale di energia.
    Rpte = [C1e[k] + C2 - We[k] - Ie[k] for k in range(1000)]
    
    # Determinazione delle variazioni nella flessibilità della domanda.
    Delta_Fe = [min(max(-Rpte[k], Min_Flex_Demand), Max_Flex_Demand) for k in range(1000)]
    
    # Calcolo del valore assoluto delle variazioni nella flessibilità della domanda.
    ABS_Delta_Fe = [abs(x) for x in Delta_Fe]
    
    # Calcolo della produzione di energia termica.
    Te = [min(max(C1e[k] + C2 + Delta_Fe[k] - Ie[k] - We[k], Min_Thermal), Max_Thermal) for k in range(1000)]
    
    # Calcolo di un altro parametro (non specificato).
    Be = [abs(C1e[k] + Delta_Fe[k] + C2 - We[k] - Ie[k] - Te[k]) for k in range(1000)]
    
    # Calcolo di diversi costi o valori relativi alla domanda, produzione termica e importazione.
    DRe = [max(abs(C1 - C1e[k]) - Band, 0) * Slope for k in range(1000)]
    DTe = [Te[k] * Price_Thermal for k in range(1000)]
    DIe = [(Ie[k] - Import) * Price_Import for k in range(1000)]
    
    # Somma dei valori calcolati.
    DRe_s = sum(DRe)
    DTe_s = sum(DTe)
    DIe_s = sum(DIe)
    ABS_Delta_Fe_s = sum(ABS_Delta_Fe)
    
    # Calcolo del prezzo della flessibilità.
    Price_Flex = (DRe_s - DIe_s - DTe_s) / ABS_Delta_Fe_s

    # Alla fine, restituisci il valore calcolato di Price_Flex
    return Price_Flex,C1,C2

# Interfaccia utente Streamlit
def main():
    st.title("Calcolatore di Price Flex")

    # Creazione dei widget per l'input dell'utente
    ToT_Demand = st.number_input("Domanda Totale di Energia", value=4700)
    Wind = st.number_input("Produzione di Energia Eolica", value=1700)
    Import = st.number_input("Quantità di Energia Importata", value=3000)
    Max_Import = st.number_input("Massima Quantità di Energia che può essere Importata", value=3000)
    Min_Import = st.number_input("Minima Quantità di Energia che può essere Importata", value=-1000)
    Thermal = st.number_input("Produzione Attuale di Energia Termica", value=0)
    Max_Thermal = st.number_input("Massima Produzione di Energia Termica Possibile", value=800)
    Min_Thermal = st.number_input("Minima Produzione di Energia Termica", value=0)
    Price_Import = st.number_input("Prezzo dell'Energia Importata", value=40)
    Price_Wind = st.number_input("Prezzo dell'Energia Eolica", value=20)
    Price_Thermal = st.number_input("Prezzo dell'Energia Termica", value=120)
    flex = st.slider("Fattore di Flessibilità", 0.01, 0.1, 0.01)
    Band = st.number_input("BAND", value=0)
    Slope = st.number_input("SLOPE", value=50)
    

    # Calcola la percentuale e visualizzala
    flex_percentuale = flex * 100
    st.write(f"Fattore di Flessibilità: {flex_percentuale}%")
    st.write(f"Intervallo di flessibilità(BAND): {Band}")
    st.write(f"Pendenza, utilizzata per calcolare l'adattamento della domanda(SLOPE): {Slope}")
 
    # Bottone per eseguire il calcolo
    if st.button("Calcola Price Flex"):
        price_flex, c1, c2 = calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal)
        st.success(f"Price Flex calcolato: {price_flex}")
        st.write(f"C1 (Domanda non flessibile): {c1}")
        st.write(f"C2 (Domanda flessibile): {c2}")

if __name__ == "__main__":
    main()
