from cProfile import label
import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from MATPLOTRC import *

START_TIME=0

class LSTMnetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.linear = nn.Sequential(
                                    nn.Linear(hidden_size,hidden_size),
                                    nn.PReLU(),
                                    nn.Linear(hidden_size,output_size))
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(seq.view(-1,seq.shape[-1],self.output_size), self.hidden)
        pred = self.linear(lstm_out)
        return pred[:,-1] 

class SEQ_LSTM:
    def __init__(self,data,test_size,step_size,hidden_size,device,split_num):
        self.data=data
        if test_size:
            self.train_set = data[:-test_size]
            self.test_set = data[-test_size:]
        else:
            self.train_set = data
            self.test_set = data[0:0]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_norm=self.scaler.fit_transform(self.train_set.reshape(-1,1)).reshape(-1)
        self.train_data=self.input_data(self.train_norm,step_size)
        self.model=LSTMnetwork(1,hidden_size,1).to(device)
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3) #lr：学习率
        self.step_size=step_size
        self.device=device
        self.split_num=split_num
    
    def input_data(self,seq,ws):  
        out = []
        L = len(seq)
        for i in range(L-ws):
            window = seq[i:i+ws]
            label = seq[i+ws:i+ws+1]
            out.append((window,label))
        out=tuple([[x[i] for x in out] for i in range(2)])
        return out

    def train(self,epochs,batchsize):
        self.model.train()
        start_time=time.time()
        sum_loss=0
        print_cycle=1000
        for epoch in range(epochs):
            for i in range(int(len(self.train_data[0])//batchsize-1e-8)+1):
                sli=slice(i*batchsize,min((i+1)*batchsize,len(self.train_data[0])))
                seq=torch.FloatTensor(np.vstack(self.train_data[0][sli])).to(self.device)
                y_train=torch.FloatTensor(np.array(self.train_data[1][sli])).reshape(-1,1).to(self.device)
                self.optimizer.zero_grad()
                self.model.hidden = (torch.zeros(1,sli.stop-sli.start,self.model.hidden_size).to(self.device),
                                     torch.zeros(1,sli.stop-sli.start,self.model.hidden_size).to(self.device))
                y_pred = self.model(seq)
                loss = self.criterion(y_pred, y_train)
                loss.backward()
                self.optimizer.step()
                sum_loss+=loss.item()
            if not (epoch+1)%print_cycle:
                print(f'Epoch: {epoch+1:2} Loss: {sum_loss/(print_cycle*len(self.train_data)):10.8f}')
                sum_loss=0
        print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    
    def predict(self,future):
        preds = self.train_norm[-self.step_size:].tolist()
        self.model.eval()
        for _ in range(future):
            seq = torch.FloatTensor(preds[-self.step_size:]).to(self.device)
            with torch.no_grad():
                self.model.hidden = (torch.zeros(1,1,self.model.hidden_size).to(self.device),
                                torch.zeros(1,1,self.model.hidden_size).to(self.device))
                preds.append(self.model(seq).item())

        true_predictions = self.scaler.inverse_transform(np.array(preds[self.step_size:]).reshape(-1,1))
        return true_predictions.reshape(-1)
    
    def plot(self,predict_size,begin_year):
        seq=torch.FloatTensor(np.vstack(self.train_data[0])).to(self.device)
        self.model.hidden = (torch.zeros(1,len(seq),self.model.hidden_size).to(self.device),
                             torch.zeros(1,len(seq),self.model.hidden_size).to(self.device))
        train_pred_norm=np.hstack([self.train_norm[:self.step_size],self.model(seq).cpu().detach().numpy().reshape(-1)])
        train_pred=self.scaler.inverse_transform(train_pred_norm.reshape(-1,1)).reshape(-1)
        test_pred=self.predict(predict_size)
        pred=np.hstack([train_pred,test_pred])
        fig,ax=plt.subplots()
        X_original=begin_year+np.linspace(0,(len(self.data)-1)/self.split_num,len(self.data))
        slic=slice(-predict_size,None,None)
        ax.plot(X_original[slic]-X_original[slic][0],self.data[slic],c=ppm_o_c,label=o_label,zorder=1)
        X_predict=begin_year+np.linspace(0,(len(self.train_set)+predict_size-1)/self.split_num,len(self.train_set)+predict_size)
        ax.scatter(X_predict[slic]-X_predict[slic][0],pred[slic],c=ppm_p_c,label=p_label,zorder=2)
        if predict_size==len(self.test_set):
            resid=pred[::self.split_num][slic]-self.data[::self.split_num][slic]
            print('reisd',resid)
            print('mse',(resid**2).mean())
        ax.set(xlabel='Steps',ylabel='Tec')
        ax.legend()
        fig.savefig('LSTM_TEC'+str(predict_size)+'.jpg',bbox_inches='tight')
        return X_predict,pred
    
    def save_model(self,path):
        torch.save(self.model,path)
    
    def load_model(self,path):
        self.model=torch.load(path)

if __name__=='__main__':
    import scipy.interpolate as spi
    torch.manual_seed(0)
    np.random.seed(0)
    data=pd.read_csv('BJ3yue.csv')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ppm=data['avgTEC']
    X=np.array(list(range(len(ppm))))
    ipo3=spi.splrep(X,ppm,k=3)
    split_num=1
    X_split=np.linspace(0,len(ppm)-1,split_num*(len(ppm)-1)+1)
    ppm_split=spi.splev(X_split,ipo3)
    np.save('ppm_split',ppm_split)
    test_size=48 #预测数

    seq_lstm_1=SEQ_LSTM(ppm_split,test_size*split_num,40,100,device,split_num) #40：窗口长度，100：神经网络宽度
    # seq_lstm_1.load_model('1lstm_model_tec'+str(test_size))
    seq_lstm_1.train(2000,2048) #1000：训练轮次，512：批量大小
    x1,y1=seq_lstm_1.plot((test_size+0)*split_num,START_TIME)
    np.save('tec_pred',np.vstack([x1,y1]))
    seq_lstm_1.save_model('1lstm_model_tec'+str(test_size))

    seq_lstm_2=SEQ_LSTM(ppm_split,test_size*split_num,40,100,device,split_num) #40：窗口长度，100：神经网络宽度
    # seq_lstm_2.load_model('2lstm_model_tec'+str(test_size))
    seq_lstm_2.train(1000,2048) #1000：训练轮次，512：批量大小
    x2,y2=seq_lstm_2.plot((test_size+0)*split_num,START_TIME)
    np.save('tec_pred',np.vstack([x2,y2]))
    seq_lstm_2.save_model('2lstm_model_tec'+str(test_size))

    p_label='LSTM'
    fig=plt.figure()
    y=ppm_split[-test_size:]
    y1=y1[-test_size:]
    y2=y2[-test_size:]
    plt.plot(y,label=o_label,c=ppm_o_c)
    plt.plot(y1,':',label=p_label+str(1),c=ppm_p_c)
    plt.plot(y2,'--',label=p_label+str(2),c=line_c)
    plt.xlabel('Steps')
    plt.ylabel('Tec')
    plt.legend()
    fig.savefig('LSTM_Three_lines.jpg',bbox_inches='tight')

    risd1=y1-y
    risd2=y2-y
    def cal_mrr(risd,fm):
        dic={'MAD':[],'RMSE':[],'RA':[]}
        mad=rmse=ra_fm=0
        for i,r in enumerate(risd):
            mad=(i*mad+np.abs(r))/(i+1)
            rmse=(i*rmse+r**2)/(i+1)
            ra_fm=(i*ra_fm+fm[i])/(i+1)
            dic['MAD'].append(mad)
            dic['RMSE'].append(rmse)
            dic['RA'].append(1-mad/ra_fm)
        return dic
    dic1=cal_mrr(risd1,y)
    dic2=cal_mrr(risd2,y)

    for key in dic1:
        fig=plt.figure()
        plt.plot(dic1[key],label=p_label+'_'+str(1),linestyle='--',c=ppm_p_c)
        plt.plot(dic2[key],label=p_label+'_'+str(2),c=line_c)
        plt.xlabel('steps')
        plt.ylabel(key)
        plt.legend()
        plt.savefig(key+'.jpg',bbox_inches='tight')