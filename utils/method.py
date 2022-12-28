import numpy as np

class DataRetriever():
    def __init__(self, buffer_size = -1, vis_window_size = 100):
        self.buffer_size = buffer_size
        self.vis_window_size = vis_window_size

        self.buffer = {'sensor': [], 'label': []}

    def isEmpty(self):
        for type in self.buffer.keys():
            if len(self.buffer[type]) == 0:
                return True
        return False
    
    def addItem(self, item):
        for key in item.keys():
            self.buffer[key].append(item[key])
            if self.buffer_size != -1:
                if(len(self.buffer[key]) > self.buffer_size):
                    self.buffer[key] = self.buffer[key][-self.buffer_size:]

    def getBuffer(self):
        return {key: np.array(self.buffer[key]) for key in self.buffer.keys()}

    def getVisWindow(self, key):
        if len(self.buffer[key]) < self.vis_window_size:
            return np.concatenate((np.zeros(self.vis_window_size - len(self.buffer[key])), self.buffer[key]))
        else:
            return np.array(self.buffer[key][-self.vis_window_size:])

class StateMethod():
    def __init__(self):
        self.state = 'init'
        self.demo_retriever = DataRetriever()
        self.global_retriever = DataRetriever(buffer_size = 200)
        self.inference_retriever = DataRetriever(buffer_size = 1000)

    def changeState(self, state):
        self.state = state

    def addItem(self, item):
        if self.state == 'demo':
            self.demo_retriever.addItem(item)
        elif self.state == 'inference':
            self.inference_retriever.addItem(item)
        elif self.state == 'train':
            self.inference_retriever.addItem(item)
        self.global_retriever.addItem(item)

    def getBuffer(self):
        if self.state == 'init' or self.demo_retriever.isEmpty():
            print("No data in buffer, please start demo or inference first")
        else:
            return self.demo_retriever.getBuffer()
        
    def getCurWindow(self, key):
        if self.state == 'demo':
            return self.demo_retriever.getBuffer()[key]
        elif self.state == 'inference':
            return self.inference_retriever.getBuffer()[key]
        else:
            return self.global_retriever.getBuffer()[key]

    def getVisWindow(self, key):
        return self.global_retriever.getVisWindow(key)