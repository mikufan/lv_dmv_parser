import timeit
import numpy as np
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def predictBatchMStep(NN_child, NN_decision, sentences_words, sentence_lens, sentences_posSeq, valency_size, trans_param): # sentences_posSeq is the same with sentences_words
    NN_child.eval()
    # NN_decision.eval()

    data = np.array([(h, d, valence)
                     for h in range(35)
                     for d in range(2)
                     for valence in range(valency_size)])

    evalDataLoader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)  # batch_size=10

    for ii, data in enumerate(evalDataLoader):
        input_pos = data[:, 0]
        direction_left = 1-data[:, 1]
        direction_right = data[:, 1]
        valence = data[:, 2]
        # output_pos = data[:, 3]

        pred_y = NN_child.forwardChd(sentences=None,  # sentences_len = [3,2] not [2, 3] sequence of decreasing
                                     sentences_len=None, h=input_pos,
                                     direction_left=direction_left, direction_right=direction_right, v=valence)
        predict_output_pos_logp = pred_y.data.numpy()
        trans_param[input_pos.numpy(),:,0,0,direction_right.numpy(), valence.numpy()] = predict_output_pos_logp

    # wf = open(predicted_dec, 'w')
    # for ii, data in enumerate(evalDecisionDataLoader):
    #     for k in range(len(data) - 1, 0, -1):  # bubble sort
    #         for j in range(0, k):
    #             if sentence_lens[data[j, 0]] < sentence_lens[data[j + 1, 0]]:
    #                 data[j], data[j + 1] = data[j + 1].clone(), data[j].clone()
    #
    #     sentence_index = data[:, 0]
    #     sts_temp = [sentences_words[idx] for idx in sentence_index]
    #     sentences_temp = []
    #     sentences_temp_len = [len(idx) for idx in sts_temp]
    #     for idx in range(len(sts_temp)):
    #         sentences_temp = sentences_temp + sts_temp[idx]
    #     input_pos = data[:, 1]
    #     direction_left = data[:, 5]
    #     direction_right = data[:, 6]
    #     valence = data[:, 7]
    #     output_pos = data[:, 3]
    #     pred_y = NN_decision.forwardChd(sentences=sentences_temp,
    #                              # sentences_len = [3,2] not [2, 3] sequence of decreasing
    #                              sentences_len=sentences_temp_len, h=input_pos,
    #                              direction_left=direction_left, direction_right=direction_right, v=valence)
    #     predict_output_pos_logp = pred_y.data.numpy()[[kk for kk in range(len(sentence_index))], output_pos.numpy()]  #pred_y.data.numpy()[:, output_pos.numpy()][0]
    #     for i in range(len(sentence_index)):
    #         idx = data[i, 0]
    #         h_idx = data[i, 2]
    #         c = data[i, 3]
    #         # dir_left = direction_left[i]
    #         dir_right = direction_right[i]
    #         val = valence[i]
    #         pred_prob = predict_output_pos_logp[i]
    #         wf.write(str(idx) + '\t' + str(h_idx) + '\t'  + str(dir_right) + '\t' + str(val) + '\t'+ str(c) + '\t' + str(pred_prob) + '\n')
    # wf.close()

def childAndDecisionANNMstep_torch(NN_child=None, NN_decision=None, nn_epouches=1, batch_size_nn=10, child_rule_samples='rule_0.txt', dec_rule_samples='rule_0.txt',
                                   sentences_words_train=None, sentence_lens_train=None, dic2Tag=None, nb_classes=None, valency_size=2, chd_nn=1, dec_nn=1,
                                   chd_lr=0.01, dec_lr=0.01):# trian and predict
    # arr_child = np.loadtxt(child_rule_str,dtype='int')
    # # arr_child = arr_chd[splitArr(arr_chd, 0)]
    # arr_decision = np.loadtxt(dec_rule_str,dtype='int')
    # # arr_decision = arr_dec[splitArr(arr_dec, 1)]
    NN_child.train(True)
    # NN_decision.train(True)

    child_rule_train = np.concatenate((np.reshape(child_rule_samples[0].astype(np.int32), (-1, 1)), np.reshape(child_rule_samples[1].astype(np.int32), (-1, 1)), np.reshape(child_rule_samples[4].astype(np.int32), (-1, 1)), np.reshape(child_rule_samples[5].astype(np.int32), (-1, 1))), axis=1)

    optimizers_child = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_child.parameters()), lr=chd_lr, weight_decay=1e-4)
    # optimizers_decision = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_decision.parameters()), lr=dec_lr, weight_decay=1e-4)
    loss_func_child = torch.nn.CrossEntropyLoss()
    loss_func_decision = torch.nn.CrossEntropyLoss()
    trainChddataloader = torch.utils.data.DataLoader(child_rule_train,batch_size = batch_size_nn, shuffle=True)  # change it to dataloader of tree_lstm with same length sts
    # trainDecDataloader = torch.utils.data.DataLoader(arr_decision, batch_size=batch_size_nn, shuffle=True)

    # train chd nn
    if chd_nn==1:
        # print("begin training:")
        running_loss = 0.0
        count = 0
        for iter in range(nn_epouches):
            # print("Epouch: " + str(iter) + "\tof Epouches " + str(nn_epouches))
            for i, data in enumerate(trainChddataloader):
                NN_child.zero_grad()
                optimizers_child.zero_grad()
                input_pos = data[:, 0].type(torch.LongTensor)
                direction_left = 1 - data[:, 2].type(torch.LongTensor)  # ??
                direction_right = data[:, 2].type(torch.LongTensor)
                valence = data[:, 3].type(torch.LongTensor)
                label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 1])).data.numpy()).type(torch.LongTensor))
                predy_extroloss = NN_child.forward_chd_train(sentences=None,
                                                    sentences_len=None, h=input_pos,
                                                    direction_left=direction_left, direction_right=direction_right, v=valence)
                loss = loss_func_child(predy_extroloss, label)
                # loss = loss + predy_extroloss[1] if predy_extroloss[1] else loss
                loss.backward()
                optimizers_child.step()
                running_loss += loss.data[0]
                count += len(data)
                if i % 1000 == 999:
                    # print('[%d, %5d] child loss:%.3f' % (iter + 1, i + 1, running_loss / count))
                    running_loss = 0
                    count = 0

    # train dec nn
    # if dec_nn==1:
    #     running_loss = 0.0
    #     count = 0
    #     for iter in range(nn_epouches):
    #         print("Epouch: " + str(iter) + "\tof Epouches " + str(nn_epouches))
    #         for i, data in enumerate(trainDecDataloader):
    #             NN_decision.zero_grad()
    #             optimizers_decision.zero_grad()
    #             for k in range(len(data) - 1, 0, -1): # bubble sort
    #                 for j in range(0, k):
    #                     if sentence_lens_train[data[j,0]] < sentence_lens_train[data[j + 1,0]]:
    #                         data[j], data[j+1] = data[j+1].clone(), data[j].clone()  # ??
    #             sentence_index = data[:,0]
    #             sts_temp = [sentences_words_train[idx] for idx in sentence_index]
    #             sentences_temp_len = [len(idx) for idx in sts_temp]
    #             sentences_temp = []
    #             for idx in range(len(sts_temp)):
    #                 sentences_temp = sentences_temp + sts_temp[idx]
    #             input_pos = data[:, 2]
    #             direction_left = data[:, 4]
    #             direction_right = data[:, 5]
    #             valence = data[:, 6]
    #             label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 3])).data.numpy())) # ??
    #             predy_extroloss = NN_decision.forward_chd_train(sentences=sentences_temp, # ?????
    #                                                 sentences_len=sentences_temp_len, h=input_pos,
    #                                                 direction_left=direction_left, direction_right=direction_right, v=valence)
    #             loss = loss_func_decision(predy_extroloss[0], label)
    #             loss = loss + predy_extroloss[1] if predy_extroloss[1] else loss
    #             loss.backward()
    #             optimizers_decision.step()
    #             running_loss += loss.data[0]
    #             count += len(data)
    #             if i % 1000 == 999:
    #                 print('[%d, %5d] decision loss :%.3f' % (iter + 1, i + 1, running_loss / count))
    #                 running_loss = 0
    #                 count = 0
