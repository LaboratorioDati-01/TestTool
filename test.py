import streamlit as st
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# Setting a fixed seed for reproducibility of random operations
random.seed(42)  # Usare un seed fisso è una buona pratica per risultati riproducibili
# Main function to calculate the flexibility price
def calcola_price_flex(ToT_Demand, Wind, Import, Max_Import, Min_Import, Thermal, Max_Thermal, Min_Thermal, flex, Band, Slope, Price_Import, Price_Wind, Price_Thermal):
    # (Insert your provided code here, adapted to use these variables)
    # Initial variable definitions for energy management
    file_path = './temp_flex.xlsx' 
    df = pd.read_excel(file_path, sheet_name=1)
    vettore_df = df.values.flatten().tolist()
    
    #Variable for test in spyder variable explorer
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
    # # Prezzi dell'energia.
    # Price_Import = 40  # Prezzo dell'energia importata.
    # Price_Wind = 20  # Prezzo dell'energia eolica.
    # Price_Thermal = 120  # Prezzo dell'energia termica.
    
    # Calculating flexible demand
    C1 = ToT_Demand * (1 - flex)  # Non-flexible demand
    C2 = ToT_Demand * flex  # Flexible demand
    Min_Flex_Demand = -C2 * 0.5  # Minimum flexible demand
    Max_Flex_Demand = C2 * 0.2  # Maximum flexible demand
  
    varianza = 300  # Desired variance
    deviazione_standard = math.sqrt(varianza)  # Calculating standard deviation

    # Creating a vector of variations based on a Gaussian distribution
    num_variazioni = 1000
    Variazione_wind = [random.gauss(Wind, deviazione_standard) for _ in range(num_variazioni)]
    
    # Ensuring realistic values for We (e.g., non-negative)
    We = [max(0, var) for var in Variazione_wind]
  
    # Calculating the vector of demand variations
    C1e = [C1*(1 + i) for i in vettore_df]

    # Calculating energy importation considering variations
    Ie = [max(min(C1e[k] + C2 - We[k], Max_Import), Min_Import) for k in range(1000)]
    
    # Calculating a parameter related to total energy production
    Rpte = [C1e[k] + C2 - We[k] - Ie[k] for k in range(1000)]
    
    # Determining variations in demand flexibility
    Delta_Fe = [min(max(-Rpte[k], Min_Flex_Demand), Max_Flex_Demand) for k in range(1000)]
    
    # Calculating the absolute value of variations in demand flexibility
    ABS_Delta_Fe = [abs(x) for x in Delta_Fe]
    
   # Calculating thermal energy production
    Te = [min(max(C1e[k] + C2 + Delta_Fe[k] - Ie[k] - We[k], Min_Thermal), Max_Thermal) for k in range(1000)]
    
    # Calculating energy balance
    Be = [abs(C1e[k] + Delta_Fe[k] + C2 - We[k] - Ie[k] - Te[k]) for k in range(1000)]
    
    # Calculating various costs or values related to demand, thermal production, and importation
    DRe = [max(abs(C1 - C1e[k]) - Band, 0) * Slope for k in range(1000)]
    DTe = [Te[k] * Price_Thermal for k in range(1000)]
    DIe = [(Ie[k] - Import) * Price_Import for k in range(1000)]
    
    DRe2 = []
    for i in range(1000):
    # Check if the index i is within the bounds of both lists
        if i < len(DRe) and i < len(C1e):
            # Calculate a new value based on elements from DRe and C1e lists
            DRe21 = (((DRe[i]) *40)+(C1*40))/C1e[i]
            # Append the calculated value to the DRe2 list
            DRe2.append(DRe21)
        else:
        # Exit the loop if the index i is out of bounds for either of the lists
            break  # Alternatively, you could use 'continue' to skip to the next iteration
    
    # Summing calculated values
    DRe_s = sum(DRe)
    DTe_s = sum(DTe)
    DIe_s = sum(DIe)
    ABS_Delta_Fe_s = sum(ABS_Delta_Fe)
    
    # Calculating the flexibility price
    Price_Flex = (DRe_s - DIe_s - DTe_s) / ABS_Delta_Fe_s
    Price_Flex = round(Price_Flex, 2)
    DRe2_media = round(sum(DRe2)/len(DRe2),2)
    C1 = round(C1,2)
    C2 = round(C2,2)
    # Returning the calculated value of Price_Flex
    return Price_Flex,C1,C2,C1e,We,Te,ABS_Delta_Fe,Ie , DRe2_media, DRe2

# Streamlit User Interface
def main():
    st.title("Flex price per hour calculator")
    Price_Flex = 0
    We = []
    Te = []
    ABS_Delta_Fe = []
    Ie = []
    C1e = []
    # Creazione dei widget per l'input dell'utente
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


    # Creating widgets for user input
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

    # Example values of 'flex' from 0.01 to 0.1
    flex_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # Example corresponding values of 'Price Flex' for different bands
    price_flex_values_BAND10 = [613.62, 353.48, 267.91, 225.46, 200.32, 183.94, 172.55, 164.17, 157.81, 153.08] 
    price_flex_values_BAND50 = [455.47, 273.37, 213.67, 184.16, 166.74, 155.45, 147.69, 142.02, 137.79, 134.71]
    
    # Creating a numeric index for the x-axis
    x_values = list(range(1000))
    # Check if all lists have at least 1000 items
    min_length = min(len(We), len(Te), len(ABS_Delta_Fe), len(Ie), len(C1e))
    if min_length < 1000:
        raise ValueError("Press Calculate Price Flex")
    
    # Calculating Total Production and Demand
    Tot_Production = [] # Production
    Tot_Demand2  = [] # Demand
    for i in x_values:
        Tot_Production2 = math.ceil(We[i]+Te[i]+ABS_Delta_Fe[i]+Ie[i])
        Tot_Production.append(Tot_Production2)
        Tot_Demand2_calc = math.ceil(C1e[i]+C2)
        Tot_Demand2.append(Tot_Demand2_calc)
    
    # Dati forniti
    a = 1073
    b = -107.1
    c = 259.1
    d = -5.558
    A = DRe2_media
    B = -6.499531908721845
    
    # Definizione delle funzioni f(x) e g(x)
    def f(x):
        return a * np.exp(b * x) + c * np.exp(d * x)
    
    def g(x):
        return A * np.exp(B * x)
    
    # Valori di x per il plot
    flex_values1 = np.linspace(0.00, 0.1, 100)
    # Calcolo dei valori di y per f(x) e g(x)
    f_values = f(flex_values1)
    g_values = g(flex_values1)
# Graph 1: Price Flex Comparison
    plt.figure(figsize=(10, 6))  # Increase the size of the graph
    plt.plot(flex_values, price_flex_values_BAND10, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='blue', label='BAND10')
    plt.plot(flex_values, price_flex_values_BAND50, marker='o', linestyle='-', markersize=7, markerfacecolor='none', markeredgecolor='orange', label='BAND50')
    plt.scatter([flex], [Price_Flex], color='red', s=50) # Highlight the current flex value
    plt.title("Price Flex Comparison")
    plt.xlabel("Flex %")
    plt.ylabel("Price Flex $/MW")
    plt.grid(True)
    plt.legend(loc='upper right')  # Modify the position of the legend
    plt.tight_layout() # Improve layout
    st.pyplot(plt)
    plt.close()
    # st.markdown(f"<h3 style='color: blue;'>Cost Demand Mean: {DRe2_media} $/MW</h3>", unsafe_allow_html=True)
# Graph 1.2: Price Offert Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(flex_values1, f_values, label='f(x)')
    plt.plot(flex_values1, g_values, label='g(x)', linestyle='--')
    plt.scatter([0.06], [f(0.06)], color='red')  # Punto di tangenza
    plt.title('Equilibrium Point')
    plt.xlabel('Flex %')
    plt.ylabel('Price Flex $/MW')
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    # st.markdown(f"<h3 style='color: blue;'>Cost Demand Mean: {DRe2_media} $/MW</h3>", unsafe_allow_html=True)
# Graph 1.3: Price Offert Comparison
    c = 259.1
    d = -5.558
    # Definizione di f(x)
    def f(x):
        return a * np.exp(b * x) + c * np.exp(d * x)

    x1, y1 = 0.00, 0.00
    x2, y2 = 0.02, 353.48
    m = (y2 - y1) / (x2 - x1)
    # Equazione della retta
    def retta(x):
        return m * (x - x1) + y1
    
    x_values1 = np.linspace(0, 0.1, 100)
    f_values = f(x_values1)
    retta_values = retta(x_values1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values1, f_values, label='f(x)')
    plt.plot(x_values1, retta_values, label='y(x)', linestyle='--')
    plt.scatter([x2], [y2], color='red')  # Punti della retta
    plt.title('Theoretical Linear Curve of  Demand')
    plt.xlabel('Flex %')
    plt.ylabel('Price Flex $/MW')
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    st.markdown(f"<h3 style='color: blue;'>Cost Demand Mean: {DRe2_media} $/MW</h3>", unsafe_allow_html=True)
# Graph 2: Cumulative Distribution Function Cost of Client    
    plt.figure(figsize=(10, 6))  
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
# Graph 3: Bar Chart of Cost Demand Mean Distribution
    # Convert DRe2 to a NumPy array
    DRe2_array = np.array(DRe2)
    # Separate the data based on the condition
    num_samples_less_than_mean = np.sum(DRe2_array < DRe2_media)
    num_samples_greater_than_mean = np.sum(DRe2_array >= DRe2_media)
    total_samples = len(DRe2_array)  
    categories = ['Lower than Average', 'Higher than Average']
    counts = [num_samples_less_than_mean, num_samples_greater_than_mean]
    fractions = [count / total_samples for count in counts]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, fractions)
    for i, fraction in enumerate(fractions):
        plt.text(i, fraction, f'{counts[i]}/{total_samples}', ha='center', va='bottom')
    plt.title('Distribution of Cost Demand Mean')
    plt.xlabel('Cost to the customer > Cost Demand Mean')
    plt.ylabel('Fraction of Samples(Probability)')  
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
# Graph 4: Cumulative Distribution Function of Balance
    plt.figure(figsize=(10, 6))  
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
# Graph 5: Bar Chart of Balance Value Distribution   
    # Separate the data into greater than zero and equal to zero
    num_samples_greater_than_zero = np.sum(Balance_array > 0)
    num_samples_equal_to_zero = np.sum(Balance_array == 0)
    total_samples = len(Balance_array)
    categories = ['Equal to Zero','Higher than Zero']
    counts = [num_samples_equal_to_zero, num_samples_greater_than_zero]
    fractions = [count / total_samples for count in counts]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, fractions, color='orange')
    for i, fraction in enumerate(fractions):
        plt.text(i, fraction, f'{counts[i]}/{total_samples}', ha='center', va='bottom')
    plt.title('Distribution of Balance Values')
    plt.xlabel('Balance Categories')
    plt.ylabel('Fraction of Samples')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()
    
if __name__ == "__main__":
    main()