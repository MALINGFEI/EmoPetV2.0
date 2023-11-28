import pylsl
import threading
import time
import numpy as np
'''
    模拟lsl发送数据
'''
class Simulated_SSVEP:
    '''
        模拟ssvep
    '''
    def __init__(self,channels=8,type='eeg',name='type',sfreq=250.0,\
                 freq_1=0.0,freq_2=0.0,\
                 freq1SNR=0.0,freq2SNR=0.0,\
                 noiseAmp=1.0):
        '''
        SSVEP模拟器
        :param channels:通道数量
        :param type: 名称
        :param name:    名称
        :param freq_1/2: 刺激关联频率
        :param freq1/2SNR； 刺激信噪比
        :param noiseAmp: 噪声幅度值
        '''
        self.channels=channels
        self.type=type
        self.name=name
        self.sfreq=sfreq
        self.freq_1=freq_1
        self.freq_2=freq_2
        self.freq1SNR=freq1SNR
        self.freq2SNR=freq2SNR
        self.noiseAmp=noiseAmp
        self.freq1Enabled=False
        self.freq2Enabled=False
        ''' 各通道属性'''
        self.chn_Ratio=np.ones(channels)    #幅值
        self.chn_Phase=np.zeros(channels)     #相位
        #   构建pylsl数据
        self.info=pylsl.stream_info(name=name,type=type,channel_count=self.channels,nominal_srate=sfreq,channel_format=pylsl.cf_float32)
        self.outlet=pylsl.stream_outlet(self.info)
        #   outlet
        self.t_Init=time.time()
        self.pushing=False
        #   缓存
        self.cache=[]
        self.cache_Len=250
    def Start(self):
        self.pushing=True
        t_pushThread=threading.Thread(target=self.t_Push,args=())
        t_pushThread.setDaemon(True)
        t_pushThread.start()

    def t_Push(self):
        '''
            循环发送数据
        :return:
        '''
        while self.pushing:
            time.sleep(1.0/self.sfreq)
            T=time.time()-self.t_Init
            raw=np.random.rand(self.channels)*self.noiseAmp     #首先生成伪随机的数据
            target1=np.zeros(self.channels,dtype=np.float64)
            target2 = np.zeros(self.channels,dtype=np.float64)
            for i in range(self.channels):
                #   构建第一个刺激
                if self.freq1Enabled:
                    target1[i]=np.sin(2*np.pi*T*self.freq_1+self.chn_Phase[i])*self.freq1SNR*self.chn_Ratio[i]
                #   第二个刺激
                if self.freq2Enabled:
                    target2[i] = np.sin(2 * np.pi * T * self.freq_2 + self.chn_Phase[i]) * self.freq2SNR*self.chn_Ratio[i]
            modified=raw+target2+target1
            #   发送数据
            self.outlet.push_sample(modified)
            #   维护缓存队列
            self.cache.append(list(modified))
            if len(self.cache)>self.cache_Len: self.cache.pop(0)
    def Stop(self):
        self.pushing=False

if __name__=='__main__':
    s=Simulated_SSVEP(freq_1=9.0,freq1SNR=5.0)
    s.freq1Enabled=True
    s.Start()
    while True:
        time.sleep(10)