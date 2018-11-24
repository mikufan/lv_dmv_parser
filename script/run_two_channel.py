import subprocess
from multiprocessing import Pool
import os
import time
import numpy as np
import sys

langsPathes = np.array(['UD_Ancient_Greek', 'UD_Ancient_Greek-PROIEL', 'UD_Arabic', 'UD_Basque', 'UD_Bulgarian',
                        'UD_Catalan', 'UD_Chinese', 'UD_Coptic', 'UD_Croatian', 'UD_Czech', 'UD_Czech-CAC',
                        'UD_Czech-CLTT', 'UD_Danish', 'UD_Dutch', 'UD_Dutch-LassySmall', 'UD_English',
                        'UD_English-ESL', 'UD_English-LinES', 'UD_Estonian', 'UD_Finnish', 'UD_Finnish-FTB',
                        'UD_French', 'UD_Galician', 'UD_Galician-TreeGal', 'UD_German', 'UD_Gothic', 'UD_Greek',
                        'UD_Hebrew', 'UD_Hindi', 'UD_Hungarian', 'UD_Indonesian', 'UD_Irish', 'UD_Italian',
                        'UD_Japanese', 'UD_Japanese-KTC', 'UD_Kazakh', 'UD_Latin', 'UD_Latin-ITTB', 'UD_Latin-PROIEL',
                        'UD_Latvian', 'UD_Norwegian', 'UD_Old_Church_Slavonic', 'UD_Persian', 'UD_Polish',
                        'UD_Portuguese',
                        'UD_Portuguese-Bosque', 'UD_Portuguese-BR', 'UD_Romanian', 'UD_Russian', 'UD_Russian-SynTagRus',
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

near_langs_list_5 = ['la-hu-la_ittb-grc_proiel-uk', 'got-la_proiel-la_ittb-cu-gl_treegal', 'id-ga-vi-cs_cltt-gl', 'lv-zh-he-tr-id',
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

near_langs_list_2 = ['grc-la', 'grc_proiel-got', 'ar-id', 'eu-lv', 'bg-no', 'ca-es_ancora', 'zh-fi_ftb', 'cop-da',
                        'hr-sk', 'cs-cs_cac', 'cs_cac-cs', 'cs_cltt-ru', 'da-sv_lines', 'nl-de', 'nl_lassysmall-de',
                        'en-en_lines', 'en_esl-en_lines', 'en_lines-sv_lines', 'et-fi', 'fi-et', 'fi_ftb-uk', 'fr-it',
                        'gl-he', 'gl_treegal-pt', 'de-nl_lassysmall', 'got-cu', 'el-ru_syntagrus', 'he-ro', 'hi-kk',
                        'hu-ru_syntagrus', 'id-pt_br', 'ga-no', 'it-es', 'ja-ug', 'ja_ktc-hi', 'kk-ta', 'la-la_ittb',
                        'la_ittb-la', 'la_proiel-cu', 'lv-ru_syntagrus', 'no-en', 'cu-la_proiel', 'fa-kk', 'pl-ru_syntagrus',
                        'pt-pt_bosque', 'pt_bosque-pt', 'pt_br-es', 'ro-sv', 'ru-cs', 'ru_syntagrus-pl', 'sa-fi_ftb', 'sk-pl',
                        'sl-sv', 'sl_sst-sv_lines', 'es-pt_br', 'es_ancora-ca', 'sv-sl', 'sv_lines-en_lines', 'swl-sa',
                        'ta-kk', 'tr-fi_ftb', 'uk-ru_syntagrus', 'ug-fa', 'vi-cs_cltt']

near_langs_list = ['grc-la-hu-la_ittb-grc_proiel-uk', 'grc_proiel-got-la_proiel-la_ittb-cu-gl_treegal', 'ar-id-ga-vi-cs_cltt-gl', 'eu-lv-zh-he-tr-id', 'bg-no-sv_lines-pl-sk-sl', 'ca-es_ancora-it-pt_bosque-pt_br-es', 'zh-fi_ftb-lv-ru_syntagrus-en-uk', 'cop-da-sv_lines-en_lines-es_ancora-gl_treegal', 'hr-sk-cs-sl-pl-cs_cac', 'cs-cs_cac-ru-sv-sl-hr', 'cs_cac-cs-sv-sl-ru-ru_syntagrus', 'cs_cltt-ru-ru_syntagrus-lv-bg-pl', 'da-sv_lines-en_lines-no-en-sl', 'nl-de-da-nl_lassysmall-es_ancora-en', 'nl_lassysmall-de-da-es_ancora-en-pt_br', 'en-en_lines-sv_lines-no-en_esl-da', 'en_esl-en_lines-en-sv_lines-da-no', 'en_lines-sv_lines-en-en_esl-da-no', 'et-fi-hr-tr-la-fi_ftb', 'fi-et-hr-tr-lv-la', 'fi_ftb-uk-lv-ru_syntagrus-en-zh', 'fr-it-es-pt_br-pt_bosque-ca', 'gl-he-gl_treegal-pt-id-pt_br', 'gl_treegal-pt-pt_bosque-es-pt_br-es_ancora', 'de-nl_lassysmall-es_ancora-nl-ca-da', 'got-cu-grc_proiel-la_proiel-la_ittb-la', 'el-ru_syntagrus-en_esl-uk-hu-sv', 'he-ro-gl_treegal-es_ancora-pt_br-gl', 'hi-kk-fa-ja_ktc-tr-ta', 'hu-ru_syntagrus-sv-el-de-da', 'id-pt_br-pt-es-gl-he', 'ga-no-gl-uk-pl-he', 'it-es-pt_bosque-pt_br-fr-ca', 'ja-ug-sa-grc-lv-fa', 'ja_ktc-hi-kk-fa-ta-ug', 'kk-ta-tr-fa-fi_ftb-lv', 'la-la_ittb-la_proiel-ru_syntagrus-lv-cs', 'la_ittb-la-la_proiel-ru_syntagrus-pl-sk', 'la_proiel-cu-got-la_ittb-la-grc_proiel', 'lv-ru_syntagrus-cs-cs_cac-ru-uk', 'no-en-da-sv_lines-en_lines-en_esl', 'cu-la_proiel-got-grc_proiel-la_ittb-la', 'fa-kk-ru_syntagrus-lv-pl-uk', 'pl-ru_syntagrus-uk-sl-sk-bg', 'pt-pt_bosque-gl_treegal-es-pt_br-it', 'pt_bosque-pt-it-pt_br-es-ca', 'pt_br-es-pt_bosque-it-pt-fr', 'ro-sv-no-pt_br-ru_syntagrus-sl', 'ru-cs-cs_cac-lv-ru_syntagrus-bg', 'ru_syntagrus-pl-uk-sl-sv-lv', 'sa-fi_ftb-lv-uk-kk-zh', 'sk-pl-hr-bg-sv_lines-uk', 'sl-sv-ru_syntagrus-da-pl-cs', 'sl_sst-sv_lines-en-en_lines-en_esl-bg', 'es-pt_br-it-pt_bosque-fr-pt', 'es_ancora-ca-pt_bosque-it-pt_br-pt', 'sv-sl-cs_cac-ro-cs-ru_syntagrus', 'sv_lines-en_lines-en-da-en_esl-no', 'swl-sa-ta-kk-ug-fi_ftb', 'ta-kk-tr-fi_ftb-sa-lv', 'tr-fi_ftb-kk-lv-hu-ru_syntagrus', 'uk-ru_syntagrus-pl-sv_lines-no-da', 'ug-fa-sa-kk-la-lv', 'vi-cs_cltt-zh-lv-ru-ru_syntagrus']


en_idx = 15

lang2i = {i:j for j,i in enumerate(langs)}

currPath = 'data/ud-treebanks-v1.4/'

idx =np.array([x for x in range(144)])


def strfind(a,b):
    return a[a.index(b):].split(' ')[1]

def Thread(arg):
    i = int(strfind(arg, '-idx'))
    print('idx:  '+str(idx))
    cmd = arg[0:arg.index('-idx')]
    print('cmd:  '+cmd)
    fname = "tuning23/ml-" +\
               "train-top5-"+str(langs[i])+\
               "-dev-"+str(langs[i])+\
               "-id-" + str(i) + ".log"
    file = open(fname, 'w')
    subprocess.call(cmd, shell=True, stdout=file)

def main():
    arglist = []
    st = int(sys.argv[1])
    print(st)
    end = int(sys.argv[2])
    print(end)
    for i in range(st, end):
        trains = 'et-la_ittb-no-fi-grc-nl-en-de-la-bg-it-hi-fr-eu-sl'#near_langs_list[i]
        dev = langs[i]
        pcmd = "python src/ml_dmv_parser.py " + '--child_neural --em_type em --cvalency 2 --do_eval --ml_comb_type 4 --stc_model_type 1 --em_iter 1  --non_neural_iter 20 --non_dscrm_iter 60 --epochs 60 --function_mask --train '\
               + trains + ' --dev ' + dev + " -idx " + str(idx[i])
        print(pcmd)
        arglist.append(pcmd)

    p = Pool(1)#20
    p.map(Thread, arglist, chunksize=1)
    p.close()
    p.join()

if __name__ == '__main__':
    main()
