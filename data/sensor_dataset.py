import numpy as np 
from torch.utils.data import Dataset
import pickle
from scipy.signal import butter,filtfilt
from utils.data_aug import DataAugmentation

class SensorDataset(Dataset):
	def __init__(self, params, labelled = False, extra_aug = False):
		self.resistance, self.label, self.time = [], [], []
		self.idx = []
		self.max_idx = 0
		self.labelled = labelled
		self.params = params
		self.extra_aug = extra_aug

	def addFile(self, file):
		with open(file, 'rb') as f:
			data = pickle.load(f)

		cur_resistance = data['sensor']
		cur_idx = np.arange(len(cur_resistance)) + self.params.window_size + self.max_idx

		self.resistance.extend(cur_resistance)
		self.idx.extend(cur_idx)

		if(self.labelled):
			cur_label = data['label']
			self.label.extend(cur_label)

	def parseData(self):
		#filtering sensor value shifting over time (comment this line if you have already filtered the data)
		self.resistance = self.butterBandpassFilter(self.resistance)
		self.minMaxNormalization()
		self.slidingTimeWindow(self.params.window_size)

	def butterBandpassFilter(self, data):
		b, a = butter(5, 0.001, btype='high')
		y = filtfilt(b, a, data)
		return y

	def minMaxNormalization(self):
		self.resistance = np.array(self.resistance)
		self.min_res = np.amin(self.resistance)
		self.max_res = np.amax(self.resistance)
		self.resistance = (self.resistance-self.min_res)/(self.max_res - self.min_res)

	def slidingTimeWindow(self, window_size):
		if(self.labelled):
			total_data = {'sensor' : [], 'label': [], 'idx': []}
		else:
			total_data = {'sensor' : [], 'idx': []}

		for i in range(len(self.resistance) - window_size +1):
			total_data['sensor'].append(self.resistance[i: i+window_size])
			total_data['idx'].append(self.idx[i+window_size-1])
			if(self.labelled):
				total_data['label'].append(self.label[i: i+window_size])

		for key in total_data.keys():
			total_data[key] = np.array(total_data[key])
			print(key, total_data[key].shape)
		self.total_data = total_data

	def __len__(self):
		return len(self.total_data['sensor'])

	def __getitem__(self,idx):
		item = {'sensor': self.total_data['sensor'][idx], 'idx': self.total_data['idx'][idx]}
		if(self.labelled):
			item['label'] = self.total_data['label'][idx]
		
		return item
