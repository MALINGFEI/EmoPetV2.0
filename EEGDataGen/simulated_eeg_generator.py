#coding=utf-8
'''
    虚拟的脑电生成器
    该程序仅用于测试系统稳定性
'''
import time
import numpy as np
import PySide2.QtWidgets
import pylsl
import threading
from PySide2.QtWidgets import QApplication,QFileDialog,QMessageBox,QCheckBox,QLabel,QComboBox,QSlider
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import QSize
import pyqtgraph as pg
from simulated_lsl_pubs import *
class Simulated_EEG_Panel:
    '''
        模拟脑电控制面板
    '''
    def __init__(self):
        '''

        '''
        self.ui=QUiLoader().load('./qt_ui_design/ssvep_lsl_data_simulator.ui')
        '''绑定界面内容及Slots'''
        self.ui.freq1.valueChanged.connect(lambda x: self.ui.freq1L.setText("{}Hz".format(x/10.0)))
        self.ui.freq2.valueChanged.connect(lambda x: self.ui.freq2L.setText("{}Hz".format(x / 10.0)))
        self.ui.freq1Amp.valueChanged.connect(lambda x: self.ui.freq1AmpL.setText("{}".format(x / 10.0)))
        self.ui.freq2Amp.valueChanged.connect(lambda x: self.ui.freq2AmpL.setText("{}".format(x / 10.0)))
        self.ui.noiseAmp.valueChanged.connect(lambda x: self.ui.noiseL.setText("{}uV".format(x / 10.0)))
        self.ui.chnList.addItems(['1','2','3','4','5','6','7','8'])
        self.ui.chnList.currentIndex=1
        self.publisher=simulated_lsl_pubs.Simulated_SSVEP()
        #   Slots
        self.ui.quitBtn.clicked.connect(lambda: self.ui.close())    #   退出按钮
        self.ui.startBtn.clicked.connect(self.StartRecord)
        self.ui.freq1Mode.clicked.connect(self.toggle_1)
        self.ui.freq2Mode.clicked.connect(self.toggle_2)
        '''     脑电原始图像数据显示  '''
        self.eeg_Canvas=pg.GraphicsLayoutWidget(show=True)
        self.eeg_Canvas.setBackground('#000')
        self.ui.eegCanvas.addWidget(self.eeg_Canvas)
        self.plot = self.eeg_Canvas.addPlot(title="原始脑电数据")
        self.curve = self.plot.plot(pen='y',label="通道均值")
        '''     显示界面    '''
        self.RefreshBtns()
        self.ui.show()

    def toggle_1(self):
        ''' 调整  频率1  激活状态'''
        self.publisher.freq1Enabled=not self.publisher.freq1Enabled
        self.RefreshBtns()
    def toggle_2(self):
        ''' 调整  频率2  激活状态'''
        self.publisher.freq2Enabled=not self.publisher.freq2Enabled
        self.RefreshBtns()
    def RefreshBtns(self):
        '''
            刷新按钮状态
        :return:
        '''
        #self.publisher=simulated_lsl_pubs.Simulated_SSVEP()
        #   按钮状态
        if not self.publisher.pushing:
            ''' '''
            self.ui.startBtn.setText("开始发布")
            self.ui.freq1Mode.setEnabled(False)
            self.ui.freq2Mode.setEnabled(False)
            return
        self.ui.freq1Mode.setEnabled(True)
        self.ui.freq2Mode.setEnabled(True)
        self.ui.startBtn.setText("停止发布")
        if self.publisher.freq1Enabled: self.ui.freq1Mode.setText('关闭')
        else: self.ui.freq1Mode.setText('开启')
        if self.publisher.freq2Enabled: self.ui.freq2Mode.setText('关闭')
        else: self.ui.freq2Mode.setText('开启')

    def StartRecord(self):
        '''
        :return:
        '''
        if self.publisher.pushing:
            self.publisher.pushing=False
            time.sleep(0.2)
            self.RefreshBtns()
            return
        self.RefreshBtns()
        self.publisher=simulated_lsl_pubs.Simulated_SSVEP(\
            channels=int(self.ui.chnList.currentText()),\
            name=self.ui.lslnameEdit.text(),\
            type=self.ui.lsltypeEdit.text(),\
            freq_1=self.ui.freq1.value(),\
            freq_2=self.ui.freq2.value(),\
            freq1SNR=self.ui.freq1Amp.value(),\
            freq2SNR=self.ui.freq2Amp.value(),\
            noiseAmp=self.ui.noiseAmp.value())
        self.publisher.Start()
        ''' 开启独立线程  '''
        self.t_Sync=threading.Thread(target=self.t_SyncParam)
        self.t_Sync.setDaemon(True)
        self.t_Sync.start()
        self.RefreshBtns()
    def t_SyncParam(self):
        '''
        同步数据的线程
        更新界面显示
        :return:
        '''
        while hasattr(self,'publisher'):
            if not self.publisher.pushing: return
            time.sleep(0.1)
            tmp=np.array(self.publisher.cache[-100:-1]).transpose()[0]
            try:
                self.curve.setData(tmp)
            except: pass
            self.publisher.freq_1=self.ui.freq1.value()
            self.publisher.freq_2=self.ui.freq2.value()
            self.publisher.freq1SNR=self.ui.freq1Amp.value()
            self.publisher.freq2SNR=self.ui.freq2Amp.value()
            self.publisher.noiseAmp=self.ui.noiseAmp.value()
if __name__=='__main__':
    app=QApplication([])
    ui=Simulated_EEG_Panel()
    app.exec_()