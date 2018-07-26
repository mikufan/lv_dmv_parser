import string, random, re
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.utils.data
import numpy as np
class AttnLSTM(nn.Module):
    def __init__(self, head_dic_size, head_dim, head_lstm_size, head_lstm_dim, valency_size, valency_dim, direct_size,
                 direct_dim, nhid, nclass, lstm_hidden_dim, dropout_p, max_length, softmax_layer_dim):
        super(AttnLSTM, self).__init__()
        self.model_type = 0  #  0 simple nn  1 lstm with atten   2 lstm  3 bag of words 4 anchored words
        self.head_dim = head_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.dropout_p = dropout_p
        self.nclass = nclass
        self.n_dec_class = 2  # stop or continue
        self.nhid = nhid

        self.lstm_layer = 1
        self.lstm_direct = 1

        if (self.lstm_direct == 1):
            self.bidirectional = False
        else:
            self.bidirectional = True

        self.hvds_dim = self.nhid + self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer  # for sts
        self.hvd_atten_sts_dim = nhid + self.lstm_hidden_dim * self.lstm_direct
        self.max_length = max_length

        self.embed_dir_left = nn.Embedding(direct_size, direct_dim)
        self.embed_dir_right = nn.Embedding(direct_size, direct_dim)
        self.head_embeddings = nn.Embedding(head_dic_size, head_dim)
        self.valency_embeddings = nn.Embedding(valency_size, valency_dim)
        self.head_lstm_embeddings = nn.Embedding(head_lstm_size, head_lstm_dim)

        pre_dir_mat = np.full((2, direct_dim), 1)
        pre_dir_mat[0] = np.full((1, direct_dim), 0)
        self.embed_dir_left.weight.data.copy_(torch.from_numpy(pre_dir_mat))
        self.embed_dir_right.weight.data.copy_(torch.from_numpy(pre_dir_mat))
        self.embed_dir_left.weight.requires_grad = False
        self.embed_dir_right.weight.requires_grad = False

        self.linear_left_direction = nn.Linear(head_dim + valency_dim, nhid)
        self.linear_right_direction = nn.Linear(head_dim + valency_dim, nhid)
        ##############
        self.linear_hvds = nn.Linear(self.hvds_dim, self.max_length)
        self.linear_hvd_atten_sts_chd = nn.Linear(self.hvd_atten_sts_dim, self.nclass)
        self.linear_hvd_atten_sts_dec = nn.Linear(self.hvd_atten_sts_dim, self.n_dec_class)
        self.dropout = nn.Dropout(self.dropout_p)

        self.linear_1_chd = nn.Linear(self.nhid, 15)
        self.linear_2_chd = nn.Linear(15, self.nclass)
        self.linear_1_dec = nn.Linear(self.nhid, self.n_dec_class)

        self.linear_1_chd_lstm = nn.Linear(self.hvd_atten_sts_dim, softmax_layer_dim)
        self.linear_2_chd_lstm = nn.Linear(softmax_layer_dim, self.nclass)
    #     ######lstm#######
    #     self.hidden = self.init_hidden(1)  # 1 here is just for init, will be changed in forward process
    #     self.lstm = nn.LSTM(head_lstm_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer,
    #                         bidirectional=self.bidirectional)  # hidden_dim // 2, num_layers=1, bidirectional=True
    #     #################
    #     if self.model_type==2:
    #         self.combine_lstmh_hid = self.nhid + self.lstm_layer*self.lstm_direct*self.lstm_hidden_dim
    #         self.model2_linear1 = nn.Linear(self.combine_lstmh_hid , 15)
    #         self.model2_linear2 = nn.Linear(15, self.nclass)
    #     #################
    #     elif self.model_type==3:
    #         self.combine_lstmh_hid = self.nhid + self.lstm_layer*self.lstm_direct*self.lstm_hidden_dim
    #         self.model2_linear1 = nn.Linear(self.combine_lstmh_hid , 15)
    #         self.model2_linear2 = nn.Linear(15, self.nclass)
    #     #################
    #     elif self.model_type==4:
    #         self.combine_lstmh_hid = self.nhid + self.lstm_layer*self.lstm_direct*self.lstm_hidden_dim
    #         self.model2_linear1 = nn.Linear(self.combine_lstmh_hid , 15)
    #         self.model2_linear2 = nn.Linear(15, self.nclass)
    #
    # def init_hidden(self, batch_init):
    #     return (autograd.Variable(torch.zeros(self.lstm_layer * self.lstm_direct, batch_init, self.lstm_hidden_dim)),
    #             # num_layers * bi-direction
    #             autograd.Variable(torch.zeros(2 * 1, batch_init, self.lstm_hidden_dim)))

    def forward(self, sentences, sentences_len, h, direction_left, direction_right, v):
        if sentences_len:
            batch_size = len(sentences_len)
            sentences_maxlen = sentences_len[0]

        emd_pos = self.head_embeddings(autograd.Variable(h))
        emd_valen = self.valency_embeddings(autograd.Variable(v))

        input_cat = torch.cat((emd_pos, emd_valen), 1)
        hid_tensor_left = F.relu(self.linear_left_direction(input_cat))  # nhid
        hid_tensor_right = F.relu(self.linear_right_direction(input_cat))
        hid_tensor = torch.mul(hid_tensor_left, self.embed_dir_left(autograd.Variable(direction_left))) + torch.mul(hid_tensor_right, self.embed_dir_right(autograd.Variable(direction_right)))
        # hid_tensor 1*nhid
        ######lstm#######

        # if self.model_type==1:
        #     self.hidden = self.init_hidden(batch_size)  # sts batch
        #     embeds = self.head_lstm_embeddings(autograd.Variable(torch.LongTensor(sentences)))
        #     sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, batch_sizes=sentences_len)
        #     sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
        #     lstm_out, self.hidden = self.lstm(sentence_in[0],
        #                                       self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
        #     sentences_lstm = torch.transpose(self.hidden[0], 0, 1).contiguous().view(batch_size, -1)  # batch_size* (num_layer*direct*hiddensize) #use h not c
        #     sentences_all_lstm = torch.transpose(lstm_out, 0, 1)
        #     atten_weight = F.softmax(self.linear_hvds(torch.cat((hid_tensor, sentences_lstm), 1)))[:,
        #                    0:sentences_maxlen]
        #     attn_applied = torch.bmm(torch.transpose(atten_weight.unsqueeze(2), 1, 2), sentences_all_lstm)  # 1*1*6
        #     return torch.cat((hid_tensor, attn_applied.squeeze(1)), 1)
        if self.model_type==0:
            return hid_tensor
        # elif self.model_type==2:
        #     self.hidden = self.init_hidden(batch_size)  # sts batch
        #     embeds = self.head_lstm_embeddings(autograd.Variable(torch.LongTensor(sentences)))
        #     sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, batch_sizes=sentences_len)
        #     sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
        #     lstm_out, self.hidden = self.lstm(sentence_in[0],
        #                                       self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
        #     sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
        #     sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
        #     return torch.cat((hid_tensor, self.dropout(sentences_all_lstm)), 1)
        # elif self.model_type==3:
        #     self.hidden = self.init_hidden(batch_size)  # sts batch
        #     embeds = self.head_lstm_embeddings(autograd.Variable(torch.LongTensor(sentences)))
        #     mask = [sentences_len[0]]
        #     for i in range(1, len(sentences_len)):
        #         mask.append(sentences_len[i] + mask[-1])
        #     stc_info = torch.sum(embeds[0:mask[0]], dim=0)
        #     for i in range(1, len(sentences_len)):
        #         stc_info = torch.cat((stc_info, torch.sum(embeds[mask[i-1]:mask[i]], dim=0)), 0)
        #     stc_info = stc_info/10   # word num /batch size
        #     return torch.cat((hid_tensor, self.dropout(stc_info)), 1)
        # elif self.model_type==4:
        #     self.hidden = self.init_hidden(batch_size)  # sts batch
        #     embeds = self.head_lstm_embeddings(autograd.Variable(torch.LongTensor(sentences)))
        #     sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, batch_sizes=sentences_len)
        #     sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
        #     lstm_out, self.hidden = self.lstm(sentence_in[0], self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
        #     sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
        #     sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
        #     return torch.cat((hid_tensor, self.dropout(sentences_all_lstm)), 1)


    def forwardChd(self, sentences, sentences_len, h, direction_left, direction_right, v):
        hid = self.forward(sentences, sentences_len, h, direction_left, direction_right, v)
        if self.model_type==0:
            return F.softmax(self.linear_2_chd(F.relu(self.linear_1_chd(hid))))
        # elif self.model_type==1:
        #     return F.softmax(self.linear_2_chd_lstm(F.relu(self.linear_1_chd_lstm(hid))))
        # elif self.model_type==2:
        #     return F.softmax(self.model2_linear2(F.relu(self.model2_linear1(hid))))
        # elif self.model_type==3:
        #     return F.softmax(self.model2_linear2(F.relu(self.model2_linear1(hid))))
        # elif self.model_type==4:
        #     return F.softmax(self.model2_linear2(F.relu(self.model2_linear1(hid))))



    def forward_chd_train(self, sentences, sentences_len, h, direction_left, direction_right, v):
        hid = self.forward(sentences, sentences_len, h, direction_left, direction_right, v)
        if self.model_type==0:
            return self.linear_2_chd(F.relu(self.linear_1_chd(hid)))
        # elif self.model_type==1:
        #     return self.linear_2_chd_lstm(F.relu(self.linear_1_chd_lstm(hid)))
        # elif self.model_type==2:
        #     return self.model2_linear2(F.relu(self.model2_linear1(hid)))
        # elif self.model_type==3:
        #     return self.model2_linear2(F.relu(self.model2_linear1(hid)))
        # elif self.model_type==4:
        #     return self.model2_linear2(F.relu(self.model2_linear1(hid)))


    def forward_stc_represetation(self, sentences, sentences_len, h, direction_left, direction_right, v):
        batch_size = len(sentences_len)
        sentences_maxlen = sentences_len[0]

        emd_pos = self.head_embeddings(autograd.Variable(h))
        emd_valen = self.valency_embeddings(autograd.Variable(v))

        input_cat = torch.cat((emd_pos, emd_valen), 1)
        hid_tensor_left = F.relu(self.linear_left_direction(input_cat))  # nhid
        hid_tensor_right = F.relu(self.linear_right_direction(input_cat))
        hid_tensor = torch.mul(hid_tensor_left, self.embed_dir_left(autograd.Variable(direction_left))) + torch.mul(hid_tensor_right, self.embed_dir_right(autograd.Variable(direction_right)))
        # hid_tensor 1*nhid
        # ######lstm#######
        #
        # if self.model_type==2:
        #     self.hidden = self.init_hidden(batch_size)  # sts batch
        #     embeds = self.head_lstm_embeddings(autograd.Variable(torch.LongTensor(sentences)))
        #     sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, batch_sizes=sentences_len)
        #     sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
        #     lstm_out, self.hidden = self.lstm(sentence_in[0],
        #                                       self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
        #     sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
        #     sentences_all_lstm = sentences_all_lstm.view(sentences_all_lstm.size()[0], -1)
        #     return self.dropout(sentences_all_lstm)

    def forward_sts_represent(self, sentences, sentences_len, h, direction_left, direction_right, v):
        stc_represent = self.forward_stc_represetation(sentences, sentences_len, h, direction_left, direction_right, v)
        return stc_represent


