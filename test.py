import streamlit as st
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# Imposta un seed specifico
random.seed(42)  # Puoi usare qualsiasi numero come seed

# Funzione principale per eseguire il calcolo
def calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal):
    # (Inserisci qui il codice che hai fornito, adattandolo per utilizzare queste variabili)
    # Definizione delle variabili iniziali relative alla gestione dell'energia.
    # Percorso del file Excel
    file_path = './temp_flex.xlsx' 
    # Carica il secondo foglio del file Excel
    # Nota: in pandas, puoi specificare il foglio tramite il nome o l'indice (iniziando da 0)
    df = pd.read_excel(file_path, sheet_name=1)  # '1' sta per il secondo foglio
    # Trasforma l'intero DataFrame in un array NumPy e poi in una lista
    vettore_df = df.values.flatten().tolist()
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
    # flex = 0.1  # Fattore di flessibilità, tra 0.1 e 1.
    # Band = 0  # Intervallo di flessibilità, compreso tra 10 e 50.
    # Slope = 50  # Pendenza, utilizzata per calcolare l'adattamento della domanda.
    
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
    
    varianza = 300  # Varianza desiderata
    deviazione_standard = math.sqrt(varianza)  # Calcolo della deviazione standard
    
    # Creazione di un vettore di variazioni basate su una distribuzione gaussiana
    num_variazioni = 1000
    Variazione_wind = [random.gauss(Wind, deviazione_standard) for _ in range(num_variazioni)]
    
    # Assicurati che i valori di We siano realistici (ad esempio, non negativi)
    We = [max(0, var) for var in Variazione_wind]
    
    # Creazione di vettori di variazioni casuali per domanda e produzione eolica.
    num_variazioni = 1000
    # Variazione_demand = [genera_variazione_demand() for _ in range(num_variazioni)]
    
    # Calcolo del vettore delle variazioni della domanda
    C1e = []
    for i in vettore_df:
        Variazione_demand_calc = C1*(1 + i) 
        C1e.append(Variazione_demand_calc)
        
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
    Price_Flex = round(Price_Flex, 2)
    # Alla fine, restituisci il valore calcolato di Price_Flex
    return Price_Flex,C1,C2

# Interfaccia utente Streamlit
def main():
    st.title("Flex Price Calculator")

    # Creazione dei widget per l'input dell'utente
    # Creating widgets for user input
    ToT_Demand = st.number_input("Total Demand of Energy (MW)", value=4700)
    Wind = st.number_input("Wind Energy Production (MW)", value=1700)
    Import = st.number_input("Imported Energy Quantity (MW)", value=3000)
    Max_Import = st.number_input("Maximum Importable Energy Quantity (MW)", value=3000)
    Min_Import = st.number_input("Minimum Importable Energy Quantity (MW)", value=-1000)
    Thermal = st.number_input("Current Thermal Energy Production (MW)", value=0)
    Max_Thermal = st.number_input("Maximum Possible Thermal Energy Production (MW)", value=800)
    Min_Thermal = st.number_input("Minimum Thermal Energy Production (MW)", value=0)
    Price_Import = st.number_input("Imported Energy Price (per unit)", value=40)
    Price_Wind = st.number_input("Wind Energy Price (per unit)", value=20)
    Price_Thermal = st.number_input("Thermal Energy Price (per unit)", value=120)
    flex = st.slider("Flexibility Factor (from 0.01 to 0.1)", 0.01, 0.1, 0.01)
    Band = st.number_input("Flexibility Range (BAND)", value=10)
    Slope = st.number_input("Slope (SLOPE)", value=50)


    # Calcola la percentuale e visualizzala
    flex_percentuale = flex * 100
    st.write(f"Flexibility Factor: {flex_percentuale}%")
    st.write(f"Flexibility Range (BAND): {Band}")
    st.write(f"Slope (used for demand adjustment): {Slope}")
    
    # Button to perform the calculation
    if st.button("Calculate Price Flex"):
        Price_Flex, c1, c2 = calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal)
        st.success(f"Calculated Price Flex: {Price_Flex}")
        st.write(f"C1 (Non-flexible Demand): {c1}")
        st.write(f"C2 (Flexible Demand): {c2}")

    # # Esempio di valori di 'flex' da 0.01 a 0.1
    flex_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    # # Esempio di corrispondenti valori di 'Price Flex'
    price_flex_values_BAND10 = [613.62, 353.48, 267.915, 225.46, 200.32, 183.94, 172.55, 164.17, 157.81, 153.08]  # Sostituisci con i tuoi valori reali
    # # Esempio di corrispondenti valori di 'Price Flex'
    price_flex_values_BAND50 = [455.47, 273.37, 213.67, 184.16, 166.74, 155.45, 147.69, 142.02, 137.79, 134.71]  # Sostituisci con i tuoi valori reali
    
 # Creazione del plot
    plt.figure()
    plt.plot(flex_values, price_flex_values_BAND10, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='blue', label='BAND10')  # Cerchi vuoti blu per BAND10
    plt.plot(flex_values, price_flex_values_BAND50, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='orange', label='BAND50')  # Marker a croce verdi per BAND50
    plt.scatter([flex], [Price_Flex], color='red', s=50)  # Pallino rosso più grande per il punto selezionato
    plt.title("Price Flex Comparison")
    plt.xlabel("Flex")
    plt.ylabel("Price Flex")
    plt.grid(True)
    plt.legend()  # Mostra la legenda

    # Visualizzazione del plot in Streamlit
    st.pyplot(plt)

if __name__ == "__main__":
    main()
