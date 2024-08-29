# SpeechFeatures
## ����
[KaldiNativeFbankSharp](https://github.com/manyeyes/KaldiNativeFbankSharp "KaldiNativeFbankSharp")�Ƕ�kaldi-native-fbank��c-api�ӿڵķ�װ��ͨ��dllexport������������⣬���԰������������źŵ�fbank������

SpeechFeatures��һ����ȫ��c#ʵ�ֵĿ⣬���ڲ�ʵ����KaldiNativeFbankSharp�����й��ܡ�

## ��;
SpeechFeatures���ڰ�����������ASR����Ŀ�п��ټ�����Ƶ��������c#����Ŀ������SpeechFeatures����ƽ̨���롢AOT���롢WebAssembly���뽫����˳����

## ���÷���
�����ο�����SpeechFeatures.OnlineFbank��Ĺ��캯����
```csharp
public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming")
// window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
```
����Ϊʾ�����룬�������Ŀ��Ҫ���ò�����dither��snip_edges��sample_rate��num_bins,window_type ����
```csharp
//�����Ŀ����
using SpeechFeatures;
//��ʼ��OnlineFbank
OnlineFbank _onlineFbank = new OnlineFbank(
                dither: 0,
                snip_edges: false,
                sample_rate: 16000,
                num_bins: 80
                );
//������Ƶsamples,��ȡ����
public float[] GetFbank(float[] samples)
{
     float[] fbanks = _onlineFbank.GetFbank(samples);
     return fbanks;
}
```


