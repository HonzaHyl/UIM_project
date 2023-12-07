import os
import pandas as pd
import numpy as np
from MyModel import MyModel
from GetScoreSepsis import GetScoreSepsis
from DataPreprocessing import DataPreprocessing


def main(filePath):
    """
    Funkce slouzi pro overovani schopnosti (predikcnich, klasifikacnich,...) navrzeneho modelu. Model bude overovan
    na skryte mnozine dat, proto je nutne dodrzet pri odevzdani tuto strukturu kodu.
    Vyuzijete funkce:
        DataPreprocessing:  Funkce pro predzpracovani dat
        MyModel:            Funkce pro vybaveni nauceneho modelu. Nauceny model se bude nacitat z externiho souboru,
                            nebude se ucit pri kazdem spusteni kodu. Veskery kod, ktery vedl k nauceni modelu,
                            vsak bude soucasti odevzdaneho projektu.
        GetScoreSepsis:     Funkce pro vyhodnoceni uspesnosti modelu

    :param filePath: Cesta k ulozenym datum (na slozku)
    :return:
        se:     Senzitivita modelu
        sp:     Specificita modelu
        acc:    Presnost modelu (acccuracy)
        fScore: F1 skore modelu
        ppv:    Pozitivni prediktivni hodnota
    """

    # 1 - Nacteni dat
    if os.path.isdir(filePath) == False:
        print("Wrong directory")

    inputData = pd.read_csv(f"{filePath}/dataSepsis.csv", delimiter=";")
    numRecords = inputData.shape[0]

    confMatrix = np.zeros((2, 2))

     # 2 - Predzpracovani dat
    preprocessedObject = DataPreprocessing(inputData.iloc[:,:-1])
   
    for idx in range(numRecords):
        targetClass = inputData.isSepsis[idx]

        # 3 - Vybaveni natrenovaneho modelu
        outputClass = MyModel(preprocessedObject.iloc[idx,:])      # Pozor, aby do modelu nevstupovala samotna trida
                                                        # objektu, pripadne dalsi nevhodne priznaky
        if outputClass == 0 or outputClass == 1:
            confMatrix[outputClass, targetClass] += 1
        else:
            print('Invalid class number. Operation aborted.')


    se, sp, acc, ppv, fScore = GetScoreSepsis(confMatrix)

    return se, sp, acc, ppv, fScore, confMatrix


se, sp, acc, ppv, fScore, confMatrix = main(r"UIM_project")    # Cesta k datum na trenovani
