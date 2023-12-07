import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def DataPreprocessing(inputData):
    """
    Funkce slouzi pro predzpracovani dat, ktera slouzi k testovani modelu. Veskery kod, ktery vedl k nastaveni
    jednotlivych kroku predzpracovani (vcetne vypoctu konstant, prumeru, smerodatnych odchylek, atp.) budou odevzdany
    spolu s celym projektem.

    :parameter inputData:
        Vstupni data, ktera se budou predzpracovavat.
    :return preprocessedData:
        Predzpracovana data na vystup
    """

    # Seznam všech sloupců, které mají být odstraněny
    drop_list = ['Temp','EtCO2','BaseExcess','HCO3','FiO2','pH',
        'PaCO2','SaO2','AST','Alkalinephos','Chloride','Bilirubin_direct',
        'Lactate','Phosphate','Bilirubin_total','TroponinI','PTT','Fibrinogen','Unit1',
        'Unit2', 'SBP', 'DBP', 'Hct', 'Age', 'Platelets', 'BUN']
    
    # Odstranění sloupců
    df = inputData.drop(columns=drop_list)

    # Výpočet interquartilového rozptylu pro každý sloupce
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Stanovení horní a dolní hranice
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Nahrazení odlehlých hodnot hodnotami NaN na základě daných hranic
    df_outliers_removed = df[(df >= lower_bound) & (df <= upper_bound)]

    # Standardizace datasetu (dojde k převedení na array)
    scaler = StandardScaler()
    finalSepsis = scaler.fit_transform(df_outliers_removed)

    # Nahrazení NaN pomocí k-nejbližších sousedů
    imputer = KNNImputer(n_neighbors=500)
    finalSepsis = imputer.fit_transform(finalSepsis)

    # Zpětné převedení na dataframe
    preprocessedData = pd.DataFrame(data= finalSepsis, columns=df.columns)

    return preprocessedData
