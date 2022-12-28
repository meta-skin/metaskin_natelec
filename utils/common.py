import os
import torch
import string 

def idx2lul(idx, device):
	lul_label = [] 
	for s in idx:
		if(s != 0):
			lul_label.append([0,1])
		else:
			lul_label.append([1,0])

	lul_label = torch.tensor(lul_label).float().to(device)
	return lul_label

def createFolder(dir):
	try:
		if not os.path.exists(dir):
			os.makedirs(dir)
	except OSError:
		print ('Error: Creating directory. ' + dir)


# key codes (MOD 97-10)
class Keys():
	def __init__(self):
		self.dict = {}
		for i, s in enumerate(string.ascii_uppercase):
			self.dict[s] = 10 + i
		self.dict['   '] = 99
		self.dict['back'] = 52
		self.dict[0] = 0
		for i, s in enumerate(range(1,10)):
			self.dict[str(s)] = s + 77 -1

		self.key_list = list(self.dict.keys())
		self.val_list = list(self.dict.values())

	def key2str(self, key):
		position = self.val_list.index(key)
		return self.key_list[position]

	def str2key(self, string):
		return self.dict[string]