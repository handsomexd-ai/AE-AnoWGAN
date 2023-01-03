%时频分析
clc;
%a = [007,014,021,028];
%b = [0,1,2,3];
%c = [DE ,FE ,BA];
%for m = 1 : 4
    %name = ['0',a(m)];
    
%data=load('辛辛那提IMS数据/2nd_test.mat');
%data=load('辛辛那提IMS数据/1st_test.mat');
%data = load('KA01/N15_M07_F10_KA01_20.mat');
%data = load('MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_1.mat');
%data = load('MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_1.mat');
data = load('CRWU/12k Drive End Bearing Fault Data/Ball/0007/B007_0.mat');
%Au=data.text_data(:,8);
%Au = data.N15_M07_F10_KA01_20.Y(7);
%Au = Au.Data(1,:);
%Au = data.bearing.gs(:,1);
%Au = data.bearing.gs(:,1);
Au = data.X118_BA_time(:,1);
Fs = 256; % 采样频率

%短时傅里叶
%[B, F, T, P] = spectrogram(Au,256,255,500,Fs);   % B是F大小行T大小列的频率峰值，P是对应的能量谱密度
%figure
%imagesc(T,F,abs(B));
%set(gca,'YDir','normal')
%ylim([0,35]);
%colorbar;
%xlabel('时间 t/s');
%ylabel('频率 f/Hz');
%title('短时傅里叶时频图');
k = 0;
% 小波
for i = 1:50240:250000
    xdata=Au(i:(i+30240));
    k=k+1;
    wavename='cmor3-3'; % 其中3－3表示Fb－Fc，Fb是带宽参数，Fc是小波中心频率 cmor是复Morlet小波
    totalscal=150; % 小波变换尺度
    Fc=centfrq(wavename); % 小波的中心频率
    c=2*Fc*totalscal;
    scals=c./(1:totalscal);
    f=scal2frq(scals,wavename,1/Fs); % 尺度转换为频率
    coefs=cwt(xdata,scals,wavename); % 连续小波系数
    t=0:1/Fs:4.5-1/Fs;
    %figure
    imagesc(t,f,abs(coefs));
    set(gca,'YDir','reverse') %设置y轴的刻度为反向（从上往下增大）
    %set(gca,'YDir','normal')
    ylim([0,100]);
    %axis off;
    colorbar;%为图形添加色标（一个颜色条）
    %set(gca,'XTickLabel',[]);  %去掉x轴刻度标签
    %set(gca,'YTickLabel',[]);  %去掉y轴刻度标签
    %set(gca,'ytick',[]); %y轴的坐标值和刻度均不显示；
    %set(gca,'xtick',[]); %x轴的坐标值和刻度均不显示；
    set(gcf,'Units','centimeter','Position',[5 5 45 20]); %设置图片大小
    set(gca,'LooseInset',get(gca,'TightInset'));
    saveas(gcf,['D:\laboratory\test\','4_KA01_22_',num2str(k),'.jpeg']);
end
%新版cwt
figure
cwt(xdata,Fs);
