import os
import pickle
import sys
from optparse import OptionParser

import numpy as np
import torch
from tqdm import tqdm

import eisner_for_dmv
import utils
from dmv_model import ldmv_model as LDMV

from torch_model.NN_module import *
# from torch_model.NN_trainer import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="train file", metavar="FILE", default="data/toy_data")
    parser.add_option("--dev", dest="dev", help="dev file", metavar="FILE", default="data/wsj10_d")

    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--batch", type="int", dest="batchsize", default=100)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/dmv.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dim", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dim", default=50)
    parser.add_option("--epochs", type="int", dest="epochs", default=5)
    parser.add_option("--tag_num", type="int", dest="tag_num", default=1)
    parser.add_option("--tag_dim", type="int", dest="tag_dim", default=5)
    parser.add_option("--dvalency", type="int", dest="d_valency", default=2)
    parser.add_option("--cvalency", type="int", dest="c_valency", default=1)
    parser.add_option("--em_type", type="string", dest="em_type", default='viterbi')

    parser.add_option("--count_smoothing", type="float", dest="count_smoothing", default=0.1)
    parser.add_option("--param_smoothing", type="float", dest="param_smoothing", default=1e-8)

    parser.add_option("--split_epoch", type="int", dest="split_epoch", default=2)
    parser.add_option("--do_split", action="store_true", dest="do_split", default=False)
    parser.add_option("--split_duration", type="int", dest="split_duration", default=5)
    parser.add_option("--split_factor", type="int", dest="split_factor", default=2)
    parser.add_option("--multi_split", action="store_true", dest="multi_split", default=False)
    parser.add_option("--em_after_split", action="store_true", dest="em_after_split", default=False)

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.01)
    parser.add_option("--outdir", type="string", dest="output", default="output")
    parser.add_option("--l2", type="float", dest="l2", default=0.0)
    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=1000)

    parser.add_option("--use_lex", action="store_true", dest="use_lex", default=False)
    parser.add_option("--prior_alpha", type="float", dest="prior_alpha", default=0.0)
    parser.add_option("--do_eval", action="store_true", dest="do_eval", default=False)
    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")
    parser.add_option("--sub_batch", type="int", dest="sub_batch_size", default=100)
    parser.add_option("--use_prior", action="store_true", dest="use_prior", default=False)
    parser.add_option("--prior_epsilon", type="float", dest="prior_epsilon", default=1)
    parser.add_option("--lex_epsilon", type="float", dest="lex_epsilon", default=1e-4)
    parser.add_option("--lex_prior_alpha", type="float", dest="lex_prior_alpha", default=0.2)

    parser.add_option("--specify_splitting", action="store_true", default=False)

    parser.add_option("--function_mask",action="store_true",default=False )

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--gold_init", action="store_true", dest="gold_init", default=False)

    parser.add_option("--e_pass", type="int", dest="e_pass", default=4)
    parser.add_option("--em_iter", type="int", dest="em_iter", default=4)

    parser.add_option("--paramem", dest="paramem", help="EM parameters file", metavar="FILE",
                      default="paramem.pickle")

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    parser.add_option("--seed", type="int", dest="seed", default=0)

    parser.add_option("--child_nn", action="store_true", default=True)
    parser.add_option("--decision_nn", action="store_true", default=True)
    parser.add_option("--root_nn", action="store_true", default=True)

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)


    def do_eval(dmv_model, w2i, pos, options):
        print "===================================="
        print 'Do evaluation on development set'
        eval_sentences = utils.read_data(options.dev, True)
        dmv_model.eval()
        eval_sen_idx = 0
        eval_data_list = list()
        devpath = os.path.join(options.output, 'eval_pred' + str(epoch + 1) + '_' + str(options.sample_idx))
        for s in eval_sentences:
            s_word, s_pos = s.set_data_list(w2i, pos)
            s_data_list = list()
            s_data_list.append(s_word)
            s_data_list.append(s_pos)
            s_data_list.append([eval_sen_idx])
            eval_data_list.append(s_data_list)
            eval_sen_idx += 1
        eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
        parse_results = {}
        for batch_id, one_batch in enumerate(eval_batch_data):
            eval_batch_words, eval_batch_pos, eval_batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                               [s[2][0] for s in one_batch]
            eval_batch_words = np.array(eval_batch_words)
            eval_batch_pos = np.array(eval_batch_pos)
            batch_score, batch_decision_score = dmv_model.evaluate_batch_score(eval_batch_words, eval_batch_pos)
            batch_parse = eisner_for_dmv.batch_parse(batch_score, batch_decision_score, dmv_model.dvalency,
                                                     dmv_model.cvalency)
            for i in range(len(eval_batch_pos)):
                parse_results[eval_batch_sen[i]] = (batch_parse[0][i], batch_parse[1][i])
        utils.eval(parse_results, eval_sentences, devpath, options.log + '_dev' + str(options.sample_idx), epoch)
        #utils.write_distribution(dmv_model)
        print "===================================="


    w2i, pos, sentences = utils.read_data(options.train, False)
    print 'Data read'
    with open(os.path.join(options.output, options.params + '_' + str(options.sample_idx)), 'w') as paramsfp:
        pickle.dump((w2i, pos, options), paramsfp)  # #Tags is 24 in WSJ??
    print 'Parameters saved'
    data_list = list()
    sen_idx = 0
    # torch.manual_seed(options.seed)
    for s in sentences:
        s_word, s_pos = s.set_data_list(w2i, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        sen_idx += 1
    batch_data = utils.construct_update_batch_data(data_list, options.batchsize)
    print 'Batch data constructed'

    lv_dmv_model = LDMV(w2i, pos, options)

    print 'Model constructed'

    lv_dmv_model.init_param(sentences)

    print 'Decoder parameters initialized'
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        lv_dmv_model.cuda(options.gpu)
    no_split = True
    splitted_epoch = 0

    print 'Pytorch model define'
    # define hyper-parameters
    nb_classes = 35
    chd_head_dim = dec_head_dim = 10
    chd_head_lstm_dim = chd_direct_dim = chd_lstm_hidden_dim = chd_softmax_layer_dim = 10
    dec_head_lstm_dim = dec_direct_dim = dec_lstm_hidden_dim = dec_softmax_layer_dim = 10
    childValency = 1
    chd_valency_dim = dec_valency_dim = 5
    chd_dropout_p = dec_dropout_p = 0.5

    if options.child_nn:
        child_model = AttnLSTM(head_dic_size=nb_classes, head_dim=chd_head_dim, head_lstm_size=nb_classes, head_lstm_dim=chd_head_lstm_dim, valency_size=childValency, valency_dim=chd_valency_dim, direct_size=2,
        direct_dim=chd_direct_dim, nhid=chd_direct_dim, nclass=nb_classes, lstm_hidden_dim=chd_lstm_hidden_dim, dropout_p=chd_dropout_p, max_length=10, softmax_layer_dim=chd_softmax_layer_dim)
        # decision_model = AttnLSTM(head_dic_size=nb_classes, head_dim=dec_head_dim, head_lstm_size=nb_classes, head_lstm_dim=dec_head_lstm_dim, valency_size=2, valency_dim=dec_valency_dim, direct_size=2,
        #                  direct_dim=dec_direct_dim, nhid=dec_direct_dim, nclass=2, lstm_hidden_dim=dec_lstm_hidden_dim, dropout_p=dec_dropout_p, max_length=10, softmax_layer_dim=dec_softmax_layer_dim)

    for epoch in range(options.epochs):
        print "\n"
        print "Training epoch " + str(epoch)
        lv_dmv_model.train()

        if splitted_epoch > 0:
            splitted_epoch += 1

        if splitted_epoch > options.split_duration and options.multi_split:
            if not options.specify_splitting:
                lv_dmv_model.split_tags(lv_dmv_model.trans_counter, options.prior_alpha,lv_dmv_model.lex_counter,
                                    options.lex_prior_alpha)
            else:
                lv_dmv_model.specified_split_tags()
            splitted_epoch = 1

        if epoch > options.split_epoch and no_split and options.do_split:
            if not options.specify_splitting:
                lv_dmv_model.split_tags(lv_dmv_model.trans_counter, options.prior_alpha, lv_dmv_model.lex_counter,
                                    options.lex_prior_alpha)
            else:
                lv_dmv_model.specified_split_tags()
            no_split = False
            splitted_epoch += 1

        if splitted_epoch > 0 and options.em_after_split:
            lv_dmv_model.em_type = "em"

        for n in range(options.em_iter):
            print 'em iteration ', n
            training_likelihood = 0.0

            trans_counter = np.zeros(
                (len(pos.keys()), len(pos.keys()), lv_dmv_model.tag_num, lv_dmv_model.tag_num, 2, options.c_valency))
            # head_pos,head_tag,direction,decision_valence,decision
            decision_counter = np.zeros((len(pos.keys()) - 1, lv_dmv_model.tag_num, 2, options.d_valency, 2))
            if options.use_lex:
                # pos,tag,word
                lex_counter = np.zeros((len(pos.keys()), lv_dmv_model.tag_num, len(w2i.keys())))
            else:
                lex_counter = None
            tot_batch = len(batch_data)
            for batch_id, one_batch in tqdm(
                    enumerate(batch_data), mininterval=2,
                    desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                batch_likelihood = 0.0
                sub_batch_data = utils.construct_batch_data(one_batch, options.sub_batch_size)
                # For each batch,put all sentences with the same length to one sub-batch
                for one_sub_batch in sub_batch_data:
                    sub_batch_words, sub_batch_pos, sub_batch_sen = [s[0] for s in one_sub_batch], \
                                                                    [s[1] for s in one_sub_batch], \
                                                                    [s[2][0] for s in one_sub_batch]
                    # E-step
                    if not options.child_nn:
                        sub_batch_likelihood = lv_dmv_model.em_e(sub_batch_pos, sub_batch_words, sub_batch_sen,
                                                             trans_counter, decision_counter, lex_counter,
                                                             lv_dmv_model.em_type)
                    else:
                        sub_batch_likelihood = lv_dmv_model.em_e(sub_batch_pos, sub_batch_words, sub_batch_sen,
                                                             trans_counter, decision_counter, lex_counter,
                                                             lv_dmv_model.em_type, child_model, None)
                    batch_likelihood += sub_batch_likelihood
                training_likelihood += batch_likelihood
            print 'Likelihood for this iteration', training_likelihood
            # M-step
            if options.use_prior:
                lv_dmv_model.apply_prior(trans_counter, lex_counter, options.prior_alpha, options.prior_epsilon,
                                         options.lex_prior_alpha, options.lex_epsilon)
            lv_dmv_model.em_m(trans_counter, decision_counter, lex_counter, child_model, None)

        if options.do_eval:
            do_eval(lv_dmv_model, w2i, pos, options)
            # Save model parameters
        with open(os.path.join(options.output, options.paramem) + str(epoch + 1) + '_' + str(options.sample_idx),
                  'w') as paramem:
            pickle.dump(
                (lv_dmv_model.trans_param, lv_dmv_model.decision_param, lv_dmv_model.lex_param, lv_dmv_model.tag_num),
                paramem)
        lv_dmv_model.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1) + '_' + str(
            options.sample_idx)))

    print 'Training finished'
