import os
import numpy as np
import xlsxwriter

idx =np.array([x for x in range(144)])

langs = np.array(['grc', 'grc_proiel', 'ar', 'eu', 'bg', 'ca', 'zh', 'cop', 'hr', 'cs', 'cs_cac', 'cs_cltt', 'da',
                  'nl', 'nl_lassysmall', 'en', 'en_esl', 'en_lines', 'et', 'fi', 'fi_ftb', 'fr', 'gl', 'gl_treegal',
                  'de', 'got', 'el', 'he', 'hi', 'hu', 'id', 'ga', 'it', 'ja', 'ja_ktc', 'kk', 'la', 'la_ittb',
                  'la_proiel',
                  'lv', 'no', 'cu', 'fa', 'pl', 'pt', 'pt_bosque', 'pt_br', 'ro', 'ru', 'ru_syntagrus', 'sa', 'sk',
                  'sl',
                  'sl_sst', 'es', 'es_ancora', 'sv', 'sv_lines', 'swl', 'ta', 'tr', 'uk', 'ug', 'vi'])

workbook = xlsxwriter.Workbook('result_23.xlsx')#worksheet.write(0, 0, 'Idx')
worksheet = workbook.add_worksheet()
currPath = 'tuning23/'



def resultofFile(fname):
    f = open(fname, 'r')
    nextlineflag = False
    print(fname)
    
    try:
        lines = f.readlines()
    finally:
        f.close()
    acc = []
    for line in lines:
        a = 'UAS is '
        if(line.startswith(a)):
            a2 = line.split(' ')[2]
            a1 = a2[0:-2]
            acc.append(a1)
    print(acc)
    return acc

for i in range(0,64):
    filePath = "tuning23/ml-" + \
               "train-top5-"+str(langs[i])+ \
               "-dev-"+str(langs[i])+ \
               "-id-" + str(i) + ".log"
    print(filePath)
    if not os.path.exists(filePath):
        print('not exist!')
        continue
    acc = resultofFile(filePath)
    print(acc)
    worksheet.write(i, 0, "train-en-"+str(langs[i]))
    for idx in range(len(acc)):
        worksheet.write(i, idx+1, str(acc[idx]))
workbook.close()
