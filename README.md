# SpeechFeatures
SpeechFeatures是一个用C#实现的库，可以快速计算音频特征，通常用于语音信号处理等场景。

## 介绍
SpeechFeatures 是一个基于 C# 实现的音频特征计算库，能够快速提取各类音频特征，广泛应用于语音信号处理等场景。该类库在框架适配方面具有良好的兼容性，支持 net45+、net60+、netcoreapp3.1 及 netstandard2.0 + 等多种环境，可实现跨平台编译、AOT 编译及 WebAssembly 编译等功能。其核心能力包括计算 kaldi fbank、whisper feature 等主流语音特征，为语音处理相关任务提供高效支持。

## 调用方法
参数参考——SpeechFeatures.OnlineFbank类的构造函数：

```csharp
/// <summary>
/// 初始化OnlineFbank类的实例，用于提取滤波器组（Fbank）特征（常用于语音信号处理等场景）
/// </summary>
/// <param name="dither">抖动值，用于在特征提取前向信号添加微小噪声，减少量化误差影响，0.0表示无抖动</param>
/// <param name="snip_edges">是否裁剪边缘帧。若为true，当信号长度不足以填充完整帧时，会丢弃不完整的边缘帧；若为false，则保留边缘帧（用零填充）</param>
/// <param name="sample_rate">输入信号的采样率（单位：Hz），需与实际信号采样率一致</param>
/// <param name="num_bins">滤波器组的数量（即输出特征的维度），决定了Fbank特征的维度大小</param>
/// <param name="frame_shift">帧移（单位：毫秒），表示相邻两帧之间的时间间隔，决定了特征的时间分辨率（默认10ms）</param>
/// <param name="frame_length">帧长（单位：毫秒），表示每帧信号的时间长度，用于计算单帧特征的原始信号窗口大小（默认25ms）</param>
/// <param name="energy_floor">能量下限值，用于限制特征计算中能量的最小值，避免数值下溢或对数计算异常（默认0f）</param>
/// <param name="debug_mel">是否启用梅尔刻度调试模式。若为true，则输出额外的调试信息或中间结果，用于验证梅尔滤波器组的正确性</param>
/// <param name="window_type">窗函数类型，用于对每帧信号进行加窗处理（默认"hamming"，即汉明窗；其他选项还有('hamming'|'hanning'|'povey'|'rectangular'|'blackman')等）</param>
/// <param name="feature_type">特征类型，指定提取的特征种类（默认"fbank"，即滤波器组特征；其他选项还有('fbank'|'whisper')）</param>
public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10f, float frame_length = 25f, float energy_floor = 0f, bool debug_mel = false, string window_type = "hamming", string feature_type = "fbank")
```

以下为示例代码，请根据项目需要配置参数：dither，snip_edges，sample_rate，num_bins,window_type,feature_type ……

```csharp
//添加项目引用
using SpeechFeatures;
//初始化OnlineFbank
OnlineFbank _onlineFbank = new OnlineFbank(
                dither: 0,
                snip_edges: false,
                sample_rate: 16000,
                num_bins: 80,
                window_type:"hanmming", // window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
                feature_type:"fbank" // feature_type (string): Type of feature ('fbank'|'whisper')
                );
//传入音频samples,获取特征
public float[] GetFbank(float[] samples)
{
     float[] fbanks = _onlineFbank.GetFbank(samples);
     return fbanks;
}
```

参考
----------
[1] https://github.com/manyeyes/KaldiNativeFbankSharp