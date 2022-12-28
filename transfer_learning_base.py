from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pytorch_metric_learning.losses import NTXentLoss

from utils.common import idx2lul
from utils.tdc_loss import TDCLoss
from params import TDCParams
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = TDCParams()
nce_loss = NTXentLoss()
bce_loss = nn.BCELoss()
tdc_loss = TDCLoss(params.tol_dist)

def transfer_train(model, train_dataloader, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    for epoch in range(params.transfer_num_epoch):
        model = transfer_train_epoch(
            epoch, model, train_dataloader, optimizer, params)
    return model

def transfer_train_epoch(epoch, encoder_model, train_dataloader, optimizer, params):
    total_train_loss = 0
    total = 0
    for i, s in enumerate(tqdm(train_dataloader)):
        sen_val = s['sensor'].float().to(device)
        class_label = s['label'][:, -1].long().to(device)
        td_label = s['idx'].float().to(device)
        encoder_model.train()

        phase_label = idx2lul(class_label, device)

        sensor = sen_val
        enc_output, phase_output = encoder_model(sensor)

        bce_loss_val = bce_loss(phase_output, phase_label)
        tdc_loss_val = tdc_loss(enc_output, td_label)

        enc_output = enc_output[class_label[:] != 0]

        sensor, class_label = sensor[class_label[:]
                                     != 0], class_label[class_label[:] != 0]
        if(len(sensor) <= 0):
            loss = tdc_loss_val + params.transfer_loss_beta * bce_loss_val
        else:
            nce_loss_val = nce_loss(enc_output, class_label)
            loss = tdc_loss_val + params.transfer_loss_alpha * \
                nce_loss_val + params.transfer_loss_beta * bce_loss_val
        total_train_loss += loss.item()
        total += sensor.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_train_loss /= len(train_dataloader)
    torch.save(encoder_model.state_dict(
    ), 'model_log/transfer/tdc_encoder_transfer_epoch_{}.pth'.format(epoch))
    print("train loss: ", total_train_loss, "epoch: ", epoch)
    return encoder_model

def data_embedding(sensor_inputs, label_idx, model):
    div = params.batch_size
    demo_embeddings = []
    demo_labels = []
    for i in range(len(sensor_inputs)//div):
        if((i+1) * div > len(sensor_inputs)):
            cur_sensor_input = sensor_inputs[i*div: -1]
            cur_idx = label_idx[i*div: -1]
        else:
            cur_sensor_input = sensor_inputs[i * div:(i+1) * div]
            cur_idx = label_idx[i * div:(i+1) * div]
        cur_sensor_input = torch.from_numpy(cur_sensor_input).float().to(device)
        enc_output, _ = model(cur_sensor_input)
        demo_embeddings.append(enc_output.detach())
        demo_labels.append(cur_idx)
    demo_embeddings = torch.cat(demo_embeddings, 0)
    demo_labels = np.concatenate(demo_labels, 0)
    return demo_embeddings, demo_labels[:, -1]

def maximum_inner_product_search(demo_embeddings, demo_labels, cur_embedding):
    embeddings = torch.cat((demo_embeddings, cur_embedding), 0)
    dotProduct = tdc_loss.distance(embeddings)[:-1,-1]
    max_idx =  torch.argmax(dotProduct)
    most_label = demo_labels[max_idx.item()]
    return most_label