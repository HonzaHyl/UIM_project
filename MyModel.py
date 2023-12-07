import joblib
 

def MyModel(inputData):
    """
    Funkce slouzi k vybaveni nauceneho modelu. Vas model bude ulozen v samostatne promenne a se spustenim se aplikuje
    na vstupni data. Tedy, model se nebude při každém spousteni znovu ucit. Ostatni kod, kterym doslo k nauceni modelu,
    take odevzdejte v ramci projektu.
    
    :param inputData:
        Vstupni data; vzdy jde o jeden objekt pro vyhodnoceni (1 pacient)
    :return outputClass:
        Vystupni trida objektu
    """
    # Vybavení modelu a provedení predikce
    loaded_model = joblib.load("UIM_project/HVH_model.joblib")
    outputClass = loaded_model.predict(inputData.values.reshape(1,-1))


    return outputClass