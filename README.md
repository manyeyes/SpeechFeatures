# SpeechFeatures
## 介绍
[KaldiNativeFbankSharp](https://github.com/manyeyes/KaldiNativeFbankSharp "KaldiNativeFbankSharp")是对kaldi-native-fbank的c-api接口的封装，通过dllexport方法导出共享库，可以帮助你获得语音信号的fbank特征。

SpeechFeatures是一个完全用c#实现的库，其内部实现了KaldiNativeFbankSharp的所有功能。

## 用途
SpeechFeatures可在包括但不限于ASR的项目中快速计算音频特征。在c#的项目中引用SpeechFeatures，跨平台编译、AOT编译、WebAssembly编译将更加顺滑。

## 调用方法
参数参考——SpeechFeatures.OnlineFbank类的构造函数：
```csharp
public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming")
// window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
```
以下为示例代码，请根据项目需要配置参数：dither，snip_edges，sample_rate，num_bins,window_type ……
```csharp
//添加项目引用
using SpeechFeatures;
//初始化OnlineFbank
OnlineFbank _onlineFbank = new OnlineFbank(
                dither: 0,
                snip_edges: false,
                sample_rate: 16000,
                num_bins: 80
                );
//传入音频samples,获取特征
public float[] GetFbank(float[] samples)
{
     float[] fbanks = _onlineFbank.GetFbank(samples);
     return fbanks;
}
```


