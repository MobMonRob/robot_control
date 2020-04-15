translation_data_20191204 = {
    "01": 0, 
    "02": 1, 
    "03": 2, 
    "04": 3, 
    "05": 4, 
    "06": 5, 
    "07": 6, 
    "08": 7, 
    "09": 8, 
    "10": 9,
    "11": 10, 
    "12": 11, 
    "13": 12, 
    "14": 13, 
    "15": 14, 
    "16": 15, 
    "17": 16, 
    "18": 17, 
    "19": 18, 
    "20": 19, 
    "21": 20, 
    "22": 21, 
    "23": 22, 
    "24": 23, 
    "25": 24, 
    "26": 25, 
    "27": 26, 
    "28": 27, 
    "29": 28, 
    "30": 29, 
    "31": 30, 
    "32": 31, 
    "33": 32, #fehlt
    "34": 33, 
    "35": 34, 
    "36": 35, 
    "37": 36, 
    "38": 37, 
    "39": 38, # einhändig
    "40": 39,
    "41": 40,
    "42": 41, #kaputt? es fehlt noch ein Datensatz?
    "B01": 42, 
    "B02": 43, 
    "B03": 44, 
    "B04": 45, #falsches Segment
    "B05": 46, 
    "B06": 47, 
    "B07": 48, 
    "B08": 49, 
    "B09": 50, #kein klarerer Bewegungsablauf
    "B10": 51,
    "B11": 52, 
    "B12": 53, 
    "B13": 54, 
    "B14": 55, 
    "B15": 56, 
    "B16": 57, # einhändig
    "B17": 58, 
    "B18": 59, 
    "B19": 60, 
    "B20": 61, 
    "B21": 62, 
    "B22": 63, 
    "B23": 64, 
    "B24": 65, 
    "B25": 66, 
    "B26": 67, 
    "B27": 68, 
    "B28": 69, 
    "B29": 70, 
    "B30": 71, 
    "B31": 72, 
    "B32": 73, 
    "B33": 74, 
    "B34": 75, 
    "B35": 76, 
    "B36": 77, 
    "B37": 78, 
    "B38": 79, 
    "B39": 80, 
    "B40": 81,
    "B41": 82, 
    } #5. csv-File ~> 7. annotation; für alle csvFiles

offset_20191204 = [{"start": 42, "end": 83, "displacement": [0.0, 0, 0, 0, 0, 0]}] # verschiebung zwischen den Messungen; auch noch die Messdaten korrigieren ~> 20cm zurück nehmen

def translation_data_20191219():
    translation = {}
    displacement = 0
    for i in range(400):
        if i == 169 or i == 210 or i == 263: #sprünge in 162-261 (2*), 261-269 (1*) #243
            displacement -= 1
        stringI = str(i)
        if len(str(i)) == 1:
            stringI = "00" + stringI
        elif len(str(i)) == 2:
            stringI = "0" + stringI
        translation[stringI] = i + displacement
    print("translation", translation)
    return translation
    #Kollisionen in Vicon: 20, 25, 58, 64(fehlt vllt), 66, 85, 102, 105, 131, 146, 162, 261, 269, 300, 312, 316, 358, 369, 371

    #Kallibrierungen: 
    # 0 Zeile 0
    # 025 ist Zeile 26 ~> failed
    # 057 ist Zeile 58 ~> failed
    # 065 ist Zeile 66 ~> failed
    # 085 ist Zeile 86 ~> failed
    # 102 ist Zeile 103 ~> failed
    # 105 ist Zeile 106 ~> failed
    # 131 ist Zeile 132 ~> failed
    # 146 ist Zeile 147 ~> failed
    # 162 ist Zeile 163 ~> failed
    # 210 failed 
    # 263 ~> failed 
    # 269 ist Zeile 267 ~> failed
    # 300 ist Zeile 298 ~> failed
    # 312 ist Zeile 310 ~> failed
    # 316 ist Zeile 314 ~> failed
    # 358 ist Zeile 356 ~> failed
    # 369 ist Zeile 367 ~> failed
    # 371 ist Zeile 369 ~> failed
    # 391 ist Zeile 389 ~> failed



    #Fehler:
    #   2 falsche auswahl, richtige auswahl vorhanden
    #  19 falsche auswahl, sehr viele optionen
    #  20 absetzen fraglich
    #  23 falsche auswahl, sehr viele optionen
    #  24 absetzen fraglich
    #  26 falscher bereich
    #  34 absetzen falscher bereich
    #  38 aufnehmen?
    #  46 aufnehmen falscher bereich
    #  64 nochmal kontrollieren
    #  75 aufnehmen falscher bereich
    #  77 kein bereich gefunden
    #  78 kein bereich gefunden
    # 114 falsche bereiche
    # 120 falscher bereich
    # 121 falscher bereich?
    # 142 falsche bereiche
    # 153 kein bereich gefunden?
    # 156 falsche bereiche gefunden
    # 173 falscher bereich (erste zwei bereiche sind zu nah)
    # 190 falsche bereiche
    # 233 falsche bereiche
    # 234 falscher bereich (erste zwei bereiche sind zu nah)
    # 245 falsche bereiche
    # 247 falsche bereiche (falscher start)
    # 255 falsche bereiche
    # 256 falscher bereich (erste zwei bereiche sind zu nah)
    # 267 falsche bereiche (falscher start)
    # 274 falsche Bereiche
    # 275 falsche Bereiche
    # 276 falsche Bereiche
    # 278/279 falsche Bereiche
    # 284 falsche Bereiche
    # 301 falsche Bereiche
    # 305 falsche Bereiche
    ## 307 kein bereich gefunden
    # 308 falsche Bereiche
    # 309 falscher bereich (erste zwei bereiche sind zu nah)
    # 314 falscher bereich (erste zwei bereiche sind zu nah)
    # 319 falscher bereich (erste zwei bereiche sind zu nah)
    # 332 falscher bereich
    # 350 falscher bereich (erste zwei bereiche sind zu nah)
    # 356 kein bereich gefunden
    # 357 falscher bereich (letzte bereiche sind zu nah)
    # 360 falscher bereich
    # 364 falscher bereich
    # 367 falscher bereich (erste zwei bereiche sind zu nah)
    #redundant: 374: [0.3, -0.3, -0.3, 0.0, 0.0, 0.0]
    # 379 falscher bereich (erste zwei bereiche sind zu nah)
    # 385 falscher bereich
