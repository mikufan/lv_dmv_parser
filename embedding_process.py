import numpy as np
import os
import os
import pickle
import sys
from optparse import OptionParser
import xlsxwriter

import numpy as np
import torch
from tqdm import tqdm

import eisner_for_dmv
import utils
from ml_dmv_model import ml_dmv_model as MLDMV
from ml_neural_m_step import m_step_model as MMODEL

# from torch_model.NN_module import *
import random


langsPathes = np.array(['UD_Ancient_Greek', 'UD_Ancient_Greek-PROIEL', 'UD_Arabic', 'UD_Basque', 'UD_Bulgarian',
                        'UD_Catalan', 'UD_Chinese', 'UD_Coptic', 'UD_Croatian', 'UD_Czech', 'UD_Czech-CAC',
                        'UD_Czech-CLTT', 'UD_Danish', 'UD_Dutch', 'UD_Dutch-LassySmall', 'UD_English',
                        'UD_English-ESL', 'UD_English-LinES', 'UD_Estonian', 'UD_Finnish', 'UD_Finnish-FTB',
                        'UD_French', 'UD_Galician', 'UD_Galician-TreeGal', 'UD_German', 'UD_Gothic', 'UD_Greek',
                        'UD_Hebrew', 'UD_Hindi', 'UD_Hungarian', 'UD_Indonesian', 'UD_Irish', 'UD_Italian',
                        'UD_Japanese', 'UD_Japanese-KTC', 'UD_Kazakh', 'UD_Latin', 'UD_Latin-ITTB',
                        'UD_Latin-PROIEL',
                        'UD_Latvian', 'UD_Norwegian', 'UD_Old_Church_Slavonic', 'UD_Persian', 'UD_Polish',
                        'UD_Portuguese',
                        'UD_Portuguese-Bosque', 'UD_Portuguese-BR', 'UD_Romanian', 'UD_Russian',
                        'UD_Russian-SynTagRus',
                        'UD_Sanskrit', 'UD_Slovak', 'UD_Slovenian', 'UD_Slovenian-SST', 'UD_Spanish',
                        'UD_Spanish-AnCora',
                        'UD_Swedish', 'UD_Swedish-LinES', 'UD_Swedish_Sign_Language', 'UD_Tamil', 'UD_Turkish',
                        'UD_Ukrainian', 'UD_Uyghur', 'UD_Vietnamese'])
langs = np.array(['grc', 'grc_proiel', 'ar', 'eu', 'bg', 'ca', 'zh', 'cop', 'hr', 'cs', 'cs_cac', 'cs_cltt', 'da',
                  'nl', 'nl_lassysmall', 'en', 'en_esl', 'en_lines', 'et', 'fi', 'fi_ftb', 'fr', 'gl', 'gl_treegal',
                  'de', 'got', 'el', 'he', 'hi', 'hu', 'id', 'ga', 'it', 'ja', 'ja_ktc', 'kk', 'la', 'la_ittb',
                  'la_proiel',
                  'lv', 'no', 'cu', 'fa', 'pl', 'pt', 'pt_bosque', 'pt_br', 'ro', 'ru', 'ru_syntagrus', 'sa', 'sk',
                  'sl',
                  'sl_sst', 'es', 'es_ancora', 'sv', 'sv_lines', 'swl', 'ta', 'tr', 'uk', 'ug', 'vi'])

langs2i = {i:j for j,i in enumerate(langs)}

near_langs_list = ['la-hu-la_ittb-grc_proiel-uk', 'got-la_proiel-la_ittb-cu-gl_treegal', 'id-ga-vi-cs_cltt-gl', 'lv-zh-he-tr-id',
                   'no-sv_lines-pl-sk-sl', 'es_ancora-it-pt_bosque-pt_br-es', 'fi_ftb-lv-ru_syntagrus-en-uk',
                   'da-sv_lines-en_lines-es_ancora-gl_treegal', 'sk-cs-sl-pl-cs_cac', 'cs_cac-ru-sv-sl-hr',
                   'cs-sv-sl-ru-ru_syntagrus', 'ru-ru_syntagrus-lv-bg-pl', 'sv_lines-en_lines-no-en-sl',
                   'de-da-nl_lassysmall-es_ancora-en', 'de-da-es_ancora-en-pt_br', 'en_lines-sv_lines-no-en_esl-da',
                   'en_lines-en-sv_lines-da-no', 'sv_lines-en-en_esl-da-no', 'fi-hr-tr-la-fi_ftb', 'et-hr-tr-lv-la',
                   'uk-lv-ru_syntagrus-en-zh', 'it-es-pt_br-pt_bosque-ca', 'he-gl_treegal-pt-id-pt_br', 'pt-pt_bosque-es-pt_br-es_ancora',
                   'nl_lassysmall-es_ancora-nl-ca-da', 'cu-grc_proiel-la_proiel-la_ittb-la', 'ru_syntagrus-en_esl-uk-hu-sv',
                   'ro-gl_treegal-es_ancora-pt_br-gl', 'kk-fa-ja_ktc-tr-ta', 'ru_syntagrus-sv-el-de-da', 'pt_br-pt-es-gl-he',
                   'no-gl-uk-pl-he', 'es-pt_bosque-pt_br-fr-ca', 'ug-sa-grc-lv-fa', 'hi-kk-fa-ta-ug', 'ta-tr-fa-fi_ftb-lv',
                   'la_ittb-la_proiel-ru_syntagrus-lv-cs', 'la-la_proiel-ru_syntagrus-pl-sk', 'cu-got-la_ittb-la-grc_proiel',
                   'ru_syntagrus-cs-cs_cac-ru-uk', 'en-da-sv_lines-en_lines-en_esl', 'la_proiel-got-grc_proiel-la_ittb-la',
                   'kk-ru_syntagrus-lv-pl-uk', 'ru_syntagrus-uk-sl-sk-bg', 'pt_bosque-gl_treegal-es-pt_br-it', 'pt-it-pt_br-es-ca',
                   'es-pt_bosque-it-pt-fr', 'sv-no-pt_br-ru_syntagrus-sl', 'cs-cs_cac-lv-ru_syntagrus-bg', 'pl-uk-sl-sv-lv',
                   'fi_ftb-lv-uk-kk-zh', 'pl-hr-bg-sv_lines-uk', 'sv-ru_syntagrus-da-pl-cs', 'sv_lines-en-en_lines-en_esl-bg',
                   'pt_br-it-pt_bosque-fr-pt', 'ca-pt_bosque-it-pt_br-pt', 'sl-cs_cac-ro-cs-ru_syntagrus', 'en_lines-en-da-en_esl-no',
                   'sa-ta-kk-ug-fi_ftb', 'kk-tr-fi_ftb-sa-lv', 'fi_ftb-kk-lv-hu-ru_syntagrus', 'ru_syntagrus-pl-sv_lines-no-da',
                   'fa-sa-kk-la-lv', 'cs_cltt-zh-lv-ru-ru_syntagrus']
near_langs_top1_list = ['la', 'got', 'id', 'lv', 'no', 'es_ancora', 'fi_ftb', 'da', 'sk', 'cs_cac', 'cs', 'ru', 'sv_lines',
                        'de', 'de', 'en_lines', 'en_lines', 'sv_lines', 'fi', 'et', 'uk', 'it', 'he', 'pt',
                        'nl_lassysmall', 'cu', 'ru_syntagrus', 'ro', 'kk', 'ru_syntagrus', 'pt_br', 'no', 'es', 'ug',
                        'hi', 'ta', 'la_ittb', 'la', 'cu', 'ru_syntagrus', 'en', 'la_proiel', 'kk', 'ru_syntagrus',
                        'pt_bosque', 'pt', 'es', 'sv', 'cs', 'pl', 'fi_ftb', 'pl', 'sv', 'sv_lines', 'pt_br', 'ca',
                        'sl', 'en_lines', 'sa', 'kk', 'fi_ftb', 'ru_syntagrus', 'fa', 'cs_cltt']

near_langs_top2_list = ['grc-la', 'grc_proiel-got', 'ar-id', 'eu-lv', 'bg-no', 'ca-es_ancora', 'zh-fi_ftb', 'cop-da',
                        'hr-sk', 'cs-cs_cac', 'cs_cac-cs', 'cs_cltt-ru', 'da-sv_lines', 'nl-de', 'nl_lassysmall-de',
                        'en-en_lines', 'en_esl-en_lines', 'en_lines-sv_lines', 'et-fi', 'fi-et', 'fi_ftb-uk', 'fr-it',
                        'gl-he', 'gl_treegal-pt', 'de-nl_lassysmall', 'got-cu', 'el-ru_syntagrus', 'he-ro', 'hi-kk',
                        'hu-ru_syntagrus', 'id-pt_br', 'ga-no', 'it-es', 'ja-ug', 'ja_ktc-hi', 'kk-ta', 'la-la_ittb',
                        'la_ittb-la', 'la_proiel-cu', 'lv-ru_syntagrus', 'no-en', 'cu-la_proiel', 'fa-kk', 'pl-ru_syntagrus',
                        'pt-pt_bosque', 'pt_bosque-pt', 'pt_br-es', 'ro-sv', 'ru-cs', 'ru_syntagrus-pl', 'sa-fi_ftb', 'sk-pl',
                        'sl-sv', 'sl_sst-sv_lines', 'es-pt_br', 'es_ancora-ca', 'sv-sl', 'sv_lines-en_lines', 'swl-sa',
                        'ta-kk', 'tr-fi_ftb', 'uk-ru_syntagrus', 'ug-fa', 'vi-cs_cltt']



def main(process):
    if process == 'sample_lang':
        arr = np.arange(len(langs))
        for _ in range(2):
            np.random.shuffle(arr)
            langarr = arr.copy()
            lang1 = [langs[x] for x in langarr]
            for i in range(8):
                print('-'.join(lang1[i*8:i*8+8]))
    elif process == 'get_e':
        loaded_file = '/home/hanwj/Code/dmv/output/dmv.model_2000'
        m_model = torch.load(loaded_file)
        for param in m_model.parameters():
            if param.size()[0]==64:
                print(type(param.data), param.size())
                np.save('/home/hanwj/Code/dmv/output/lang_embedding', param.data.numpy())
    elif process == 'near2lang':
        emb = np.load('/home/hanwj/Code/dmv/output/lang_embedding.npy')  # 64, 20
        distances = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                distances[i][j] = np.linalg.norm(emb[i]-emb[j])
        lang_s = 'lv'
        lang_d = {distances[langs2i[lang_s]][i]: langs[i] for i in range(64)}
        print(lang_d)
        # sorted(lang_d)
        print([(k,lang_d[k]) for k in sorted(lang_d.keys())])

    elif process == 'allnear2lang':
        emb = np.load('/home/hanwj/Code/dmv/output/lang_embedding.npy')  # 64, 20
        distances = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                distances[i][j] = np.linalg.norm(emb[i] - emb[j])
        near_lang_list = []
        for meta_lang in langs:
            lang_d = {distances[langs2i[meta_lang]][i]: langs[i] for i in range(64)}
            nearby = [(k, lang_d[k]) for k in sorted(lang_d.keys())][0:6]
            print nearby
            near_lang_list.append('-'.join([i[1] for i in nearby]))
        print near_lang_list  # the result in line 48
    elif process == 'save2xlxs':
        emb = np.load('/home/hanwj/Code/dmv/output/lang_embedding.npy')  # 64, 20
        distances = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                distances[i][j] = np.linalg.norm(emb[i]-emb[j])

        workbook = xlsxwriter.Workbook('/home/hanwj/Code/dmv/output/result_10.xlsx')  # worksheet.write(0, 0, 'Idx')
        worksheet = workbook.add_worksheet()
        for i in range(64):
            for j in range(64):
                worksheet.write(i, j, str(distances[i][j]))
        workbook.close()

if __name__ == '__main__':
    process = 'allnear2lang'#'near2lang'#'get_e'#'sample_lang'
    main(process)

