import streamlit as st
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np

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
    # ToT_Demand = 4700  # Domanda totale di energia.
    # Wind = 1700  # Produzione di energia eolica.
    
    # # Parametri per l'importazione di energia.
    # Import = 3000  # Quantità di energia importata.
    # Max_Import = 3000  # Massima quantità di energia che può essere importata.
    # Min_Import = -1000  # Minima quantità di energia che può essere importata (negativo indica esportazione).
    
    # # Parametri per la produzione di energia termica.
    # Thermal = 0  # Produzione attuale di energia termica.
    # Max_Thermal = 800  # Massima produzione di energia termica possibile.
    # Min_Thermal = 0  # Minima produzione di energia termica (non può essere negativa).
    
    # Parametri di flessibilità della domanda.
    # flex = 0.1  # Fattore di flessibilità, tra 0.1 e 1.
    # Band = 0  # Intervallo di flessibilità, compreso tra 10 e 50.
    # Slope = 50  # Pendenza, utilizzata per calcolare l'adattamento della domanda.
    
    # Calcolo della domanda flessibile.
    C1 = ToT_Demand * (1 - flex)  # Domanda non flessibile.
    C2 = ToT_Demand * flex  # Domanda flessibile.
    Min_Flex_Demand = -C2 * 0.5  # Minima domanda flessibile.
    Max_Flex_Demand = C2 * 0.2  # Massima domanda flessibile.
    
    # # Prezzi dell'energia.
    # Price_Import = 40  # Prezzo dell'energia importata.
    # Price_Wind = 20  # Prezzo dell'energia eolica.
    # Price_Thermal = 120  # Prezzo dell'energia termica.
    
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
    DRe2 = []
    for i in range(1000):
    # Check if the index i is within the bounds of both lists
        if i < len(DRe) and i < len(C1e):
            DRe21 = (((DRe[i]) *40)+(C1*40))/C1e[i]
            # DRe21 = ((DRe[i]) *(We[i]*Price_Wind+Ie[i]*Price_Import+Te[i]*Price_Thermal/C1+C2)+(C1*(We[i]*Price_Wind+Ie[i]*Price_Import+Te[i]*Price_Thermal/C1+C2)))/C1e[i]
            DRe2.append(DRe21)
        else:
        # Handle the case where i is out of bounds for one of the lists
            break  # or continue, depending on your needs
    # Somma dei valori calcolati.
    DRe_s = sum(DRe)
    DTe_s = sum(DTe)
    DIe_s = sum(DIe)
    ABS_Delta_Fe_s = sum(ABS_Delta_Fe)
    
    # Calcolo del prezzo della flessibilità.
    Price_Flex = (DRe_s - DIe_s - DTe_s) / ABS_Delta_Fe_s
    Price_Flex = round(Price_Flex, 2)
    DRe2_media = round(sum(DRe2)/len(DRe2),2)
    C1 = round(C1,2)
    C2 = round(C2,2)
    # Alla fine, restituisci il valore calcolato di Price_Flex
    return Price_Flex,C1,C2,C1e,We,Te,ABS_Delta_Fe,Ie , DRe2_media, DRe2
# Interfaccia utente Streamlit
def main():
    st.title("Flex price per hour calculator")
    Price_Flex = 0
    We = []
    Te = []
    ABS_Delta_Fe = []
    Ie = []
    C1e = []
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
    Price_Import = st.number_input("Imported Energy Price ($)", value=40)
    Price_Wind = st.number_input("Wind Energy Price ($)", value=20)
    Price_Thermal = st.number_input("Thermal Energy Price ($)", value=120)
    flex = st.slider("Flexibility Factor 0%-10% (from 0.01 to 0.1)", 0.01, 0.1, 0.01)
    Band = st.number_input("Flexibility Range (BAND)", value=10)
    Slope = st.number_input("Slope (SLOPE)", value=50)


    # Calcola la percentuale e visualizzala
    flex_percentuale = flex * 100
    st.write(f"Flexibility Factor: {flex_percentuale}%")
    st.write(f"Flexibility Range (BAND): {Band}")
    st.write(f"Slope (used for demand adjustment): {Slope}")
    
    # Button to perform the calculation
    if st.button("Calculate Price Flex"):
        Price_Flex, C1, C2, C1e, We, Te, ABS_Delta_Fe, Ie, DRe2_media, DRe2 = calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal)
        st.success(f"Calculated Price Flex: {Price_Flex} $/MW")
        st.write(f"C1 (Non-flexible Demand): {C1}MW")
        st.write(f"C2 (Flexible Demand): {C2}MW")
        st.write(f"Cost Demand Mean : {DRe2_media}$/MW")

    # # Esempio di valori di 'flex' da 0.01 a 0.1
    flex_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    
    # # Esempio di corrispondenti valori di 'Price Flex'
    price_flex_values_BAND10 = [613.62, 353.48, 267.915, 225.46, 200.32, 183.94, 172.55, 164.17, 157.81, 153.08]  # Sostituisci con i tuoi valori reali
    # # Esempio di corrispondenti valori di 'Price Flex'
    price_flex_values_BAND50 = [455.47, 273.37, 213.67, 184.16, 166.74, 155.45, 147.69, 142.02, 137.79, 134.71]  # Sostituisci con i tuoi valori reali
    
    
    # Creazione di un indice numerico per l'asse x
    x_values = list(range(1000))  # 1000 punti da 0 a 999
    # Check if all lists have at least 1000 items
    min_length = min(len(We), len(Te), len(ABS_Delta_Fe), len(Ie), len(C1e))
    if min_length < 1000:
        raise ValueError("Press Calculate Price Flex")
    
    Tot_Production = [] #Production
    Tot_Demand2  = [] #Domanda
    for i in x_values:
        Tot_Production2 = math.ceil(We[i]+Te[i]+ABS_Delta_Fe[i]+Ie[i])
        Tot_Production.append(Tot_Production2)
        Tot_Demand2_calc = math.ceil(C1e[i]+C2)
        Tot_Demand2.append(Tot_Demand2_calc)
    
    
 # Grafico 1: Price Flex Comparison
    plt.figure(figsize=(10, 6))  # Aumenta le dimensioni del grafico
    plt.plot(flex_values, price_flex_values_BAND10, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='blue', label='BAND10')
    # plt.plot(flex_values, CurvefitBand10, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='green', label='BAND10-FIT')
    plt.plot(flex_values, price_flex_values_BAND50, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='orange', label='BAND50')
    plt.scatter([flex], [Price_Flex], color='red', s=50)
    plt.title("Price Flex Comparison")
    plt.xlabel("Flex %")
    plt.ylabel("Price Flex $/MW")
    plt.grid(True)
    plt.legend(loc='upper right')  # Modifica la posizione della legenda
    plt.tight_layout()  # Migliora il layout
    st.pyplot(plt)
    
      # Dividi i dati in 10 gruppi da 100 punti ciascuno
    num_groups = 10
    group_size = 100  # o min_length // num_groups per dividere equamente i punti che hai
    num_cols = 2
    
    # Calcola il numero totale di righe necessarie
    num_rows = math.ceil(num_groups / num_cols)
    plt.close()
    
    plt.figure(figsize=(10, 6))  # Aumenta le dimensioni del grafico
        # Sort the data in ascending order
    data_sorted = np.sort(DRe2)
    
    # Calculate the cumulative probabilities, from 0 to 1
    cdf_values = np.arange(len(DRe2)) / float(len(DRe2) - 1)
    
    # Create the CDF plot
    plt.plot(data_sorted, cdf_values,marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='blue')
    
    # Set the title and labels
    plt.title("Cumulative Distribution Function Cost of Client")
    plt.xlabel("Cost to the customer [$/MWh]")
    plt.ylabel("CDF Cost of Client Value (Probability)")
    
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    # Convert DRe2 to a NumPy array
    DRe2_array = np.array(DRe2)
    # Separate the data based on the condition
    num_samples_less_than_mean = np.sum(DRe2_array < DRe2_media)
    num_samples_greater_than_mean = np.sum(DRe2_array >= DRe2_media)
    
    # Calcola il numero totale dei campioni
    total_samples = len(DRe2_array)
    
    # Crea i dati per il grafico a barre
    categories = ['Lower than Average', 'Higher than Average']
    counts = [num_samples_less_than_mean, num_samples_greater_than_mean]
    fractions = [count / total_samples for count in counts]
    
    # Crea il grafico a barre
    plt.figure(figsize=(10, 6))
    plt.bar(categories, fractions)
    
    # Aggiungi le etichette con i conteggi effettivi
    for i, fraction in enumerate(fractions):
        plt.text(i, fraction, f'{counts[i]}/{total_samples}', ha='center', va='bottom')
    
    # Aggiungi etichette e titolo
    plt.title('Distribution of Cost Demand Mean ')
    plt.xlabel('Cost to the customer [$/MWh]')
    plt.ylabel('Fraction of Samples(Probability)')  

    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    plt.figure(figsize=(10, 6))  # Aumenta le dimensioni del grafico
        # Sort the data in ascending order
    Balance = []
    for i in x_values:
        Balance2 = Tot_Demand2[i] - Tot_Production[i]
        Balance.append(Balance2)    
     
    Balance_array = np.array(Balance)
    
    data_sorted = np.sort(Balance)
    
    # Calculate the cumulative probabilities, from 0 to 1
    cdf_values = np.arange(len(Balance)) / float(len(Balance) - 1)
    
    # Create the CDF plot
    plt.plot(data_sorted, cdf_values,marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='orange')
    
    # Set the title and labels
    plt.title("Cumulative Distribution Function Balance")
    plt.xlabel("MWh")
    plt.ylabel("CDF Balance Value (Probability)")
    
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    
    # Separate the data into greater than zero and equal to zero
    num_samples_greater_than_zero = np.sum(Balance_array > 0)
    num_samples_equal_to_zero = np.sum(Balance_array == 0)
    
    # Calcola il numero totale dei campioni
    total_samples = len(Balance_array)
    
    # Crea i dati per il grafico a barre
    categories = ['Equal to Zero','Higher than Zero']
    counts = [num_samples_equal_to_zero, num_samples_greater_than_zero]
    fractions = [count / total_samples for count in counts]
    
    # Crea il grafico a barre
    plt.figure(figsize=(10, 6))
    plt.bar(categories, fractions, color='orange')
    
    # Aggiungi le etichette con i conteggi effettivi
    for i, fraction in enumerate(fractions):
        plt.text(i, fraction, f'{counts[i]}/{total_samples}', ha='center', va='bottom')
    
    # Aggiungi etichette e titolo
    plt.title('Distribution of Balance Values')
    plt.xlabel('Balance Categories')
    plt.ylabel('Fraction of Samples')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    
if __name__ == "__main__":
    main()


# for group in range(num_groups):
#     plt.close()
#     plt.figure(figsize=(10, 6))
#     # Calcola gli indici per l'attuale gruppo di dati
#     start_idx = group * group_size
#     end_idx = start_idx + group_size

#     # Estrai i dati per l'attuale gruppo
#     x_values_group = range(start_idx, end_idx)
#     Tot_Production_group = [math.ceil(We[i] + Te[i] + ABS_Delta_Fe[i] + Ie[i]) for i in x_values_group]
#     Tot_Demand2_group = [math.ceil(C1e[i] + C2) for i in x_values_group]
#     DRe2_group = [math.ceil(DRe2[i]) for i in x_values_group]
#     # Crea una curva separata per ciascun gruppo di dati
#     # plt.plot(x_values_group, Tot_Demand2_group,  linewidth=2, label=f'Demand Group {group+1}')
#     # plt.plot(x_values_group, Tot_Production_group, label=f'Production Group {group+1}')
#     plt.plot(x_values_group, DRe2_group, label=f'Cost Group {group+1}')
#     # st.pyplot(plt)

#     plt.title(f"Cost Demand {group+1}")
#     plt.xlabel("Round")
#     plt.ylabel("Price ($/MW)")
#     plt.grid(True)
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     st.pyplot(plt)
#     # Numero di colonne per i subplot

# # Crea la figura e gli assi per i subplot
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))

# for group in range(num_groups):
#     # Calcola gli indici per la riga e la colonna correnti
#     row = group // num_cols
#     col = group % num_cols

#     # Ottieni l'asse corrente
#     ax = axs[row, col]

#     # Calcola gli indici per l'attuale gruppo di dati
#     start_idx = group * group_size
#     end_idx = start_idx + group_size
#     x_values_group = range(start_idx, end_idx)
    
#     # Estrai e calcola i dati
#     # (qui vanno le tue operazioni sui dati)
#     # Tot_Production_group = [math.ceil(We[i] + Te[i] + ABS_Delta_Fe[i] + Ie[i]) for i in x_values_group]
#     # Tot_Demand2_group = [math.ceil(C1e[i] + C2) for i in x_values_group]
#     DRe2_group = [math.ceil(DRe2[i]) for i in x_values_group]
#     # Crea il grafico nel subplot corrispondente
#     # ax.plot(x_values_group, Tot_Demand2_group, linewidth=2, label=f'Demand Group {group+1}')
#     # ax.plot(x_values_group, Tot_Production_group, label=f'Production Group {group+1}')
#     ax.plot(x_values_group, DRe2_group, label=f'Cost Group {group+1}')

#     # Imposta titolo, etichette e altre opzioni per l'asse corrente
#     ax.set_title(f"DCost Demand {group+1}")
#     ax.set_xlabel("Round")
#     ax.set_ylabel("Energy (MW)")
#     ax.grid(True)
#     ax.legend(loc='upper right')
    
# plt.tight_layout()
# st.pyplot(plt)    

#  # Crea la figura e gli assi per i subplot
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
# for group in range(num_groups):
#     # Calcola gli indici per la riga e la colonna correnti
#     row = group // num_cols
#     col = group % num_cols

#     # Ottieni l'asse corrente
#     ax = axs[row, col]

#     # Calcola gli indici per l'attuale gruppo di dati
#     start_idx = group * group_size
#     end_idx = start_idx + group_size
#     x_values_group = range(start_idx, end_idx)
    
#     # Estrai e calcola i dati
#     # (qui vanno le tue operazioni sui dati)
#     Tot_Production_group = [math.ceil(We[i] + Te[i] + ABS_Delta_Fe[i] + Ie[i]) for i in x_values_group]
#     Tot_Demand2_group = [math.ceil(C1e[i] + C2) for i in x_values_group]
#     # DRe2_group = [math.ceil(DRe2[i]) for i in x_values_group]
#     # Crea il grafico nel subplot corrispondente
#     ax.plot(x_values_group, Tot_Demand2_group, linewidth=2, label=f'Demand Group {group+1}')
#     ax.plot(x_values_group, Tot_Production_group, label=f'Production Group {group+1}')
#     # ax.plot(x_values_group, DRe2_group, label=f'Cost Group {group+1}')

#     # Imposta titolo, etichette e altre opzioni per l'asse corrente
#     ax.set_title(f"Demand and Production {group+1}")
#     ax.set_xlabel("Round")
#     ax.set_ylabel("Energy (MW)")
#     ax.grid(True)
#     ax.legend(loc='upper right')
    
# plt.tight_layout()
# st.pyplot(plt)