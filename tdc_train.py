import torch
from torch.utils.data import DataLoader
from torch import optim
from data.sensor_dataset import SensorDataset
from utils.data_aug import DataAugmentation
from model.transformer_encoder import TransformerEncoder
from utils.tdc_loss import TDCLoss
from params import TDCParams
from tqdm import tqdm
import os

torch.manual_seed(0)

def train(epoch, encoder_model, train_dataloader):
    global tdc_loss, optimizer
    total_train_loss, total, = 0, 0 

    pbar = tqdm(train_dataloader)
    for _, s in enumerate(pbar):
        sen_val = s['sensor'].float().to(device)
        td_label = s['idx'].float().to(device)

        encoder_model.train()

        sensor_1, sensor_2 = data_augmentation.jittering(
            sen_val), data_augmentation.jittering(sen_val)
        td_label = torch.cat([td_label, td_label], 0)
        sensor = torch.cat([sensor_1, sensor_2], 0)

        enc_output, _ = encoder_model(sensor)

        loss = tdc_loss(enc_output, td_label)
        total_train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        total += sensor.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_train_loss /= len(train_dataloader)
    if(epoch%100==0):
        torch.save(encoder_model.state_dict(), 'model_log/tdc_encoder_DS_epoch_{}.pth'.format(epoch))
    print("train loss: ", total_train_loss, "epoch: ", epoch)


def main(params=None):
    global device, data_augmentation, tdc_loss, optimizer
    if params is None:
        params = TDCParams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_augmentation = DataAugmentation(
        device=device, params = params, window_size=params.window_size)

    tdc_loss = TDCLoss(params.tol_dist)

    sensor_encoder = TransformerEncoder(
        d_model=params.d_model,
        d_embedding=params.d_embedding,
        n_layers=params.n_layers,
        n_head=params.n_head,
        window_size=params.window_size,
    ).to(device)
    
    sensorDataset = SensorDataset(params, extra_aug=True)
    for file_name in os.listdir(params.train_data_dir):
        file_path = os.path.join(params.train_data_dir, file_name) 
        if os.path.isfile(file_path):
            sensorDataset.addFile(file_path)
    sensorDataset.parseData()

    train_dataloader = DataLoader(
        sensorDataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

    optimizer = optim.Adam(sensor_encoder.parameters(), lr=params.lr)

    for epoch in range(params.num_epoch):
        train(epoch, sensor_encoder,  train_dataloader)

if __name__ == "__main__":
    main()
