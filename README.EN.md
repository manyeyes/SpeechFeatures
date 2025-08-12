# SpeechFeatures
SpeechFeatures is a library implemented in C# that can quickly compute audio features, typically used in scenarios such as speech signal processing.

## Introduction
SpeechFeatures is an audio feature computation library based on C# implementation, capable of quickly extracting various audio features and widely applied in scenarios like speech signal processing. This library boasts excellent compatibility in terms of framework adaptation, supporting multiple environments including .NET 4.5+, .NET 6.0+, .NET Core 3.1, and .NET Standard 2.0+. It enables functionalities such as cross-platform compilation, AOT compilation, and WebAssembly compilation. Its core capabilities include computing mainstream speech features like kaldi fbank and whisper feature, providing efficient support for speech processing-related tasks.

## Calling Method
Parameter reference - Constructor of the SpeechFeatures.OnlineFbank class:

```csharp
/// <summary>
/// Initializes an instance of the OnlineFbank class for extracting filter bank (Fbank) features (commonly used in scenarios such as speech signal processing)
/// </summary>
/// <param name="dither">Dither value, used to add slight noise to the signal before feature extraction to reduce the impact of quantization errors; 0.0 means no dithering</param>
/// <param name="snip_edges">Whether to snip edge frames. If true, incomplete edge frames will be discarded when the signal length is insufficient to fill a complete frame; if false, edge frames will be retained (padded with zeros)</param>
/// <param name="sample_rate">Sampling rate of the input signal (in Hz), which must be consistent with the actual signal sampling rate</param>
/// <param name="num_bins">Number of filter banks (i.e., the dimension of output features), determining the dimension size of Fbank features</param>
/// <param name="frame_shift">Frame shift (in milliseconds), representing the time interval between adjacent frames, determining the temporal resolution of features (default 10ms)</param>
/// <param name="frame_length">Frame length (in milliseconds), representing the time length of each frame of signal, used to calculate the original signal window size for single-frame feature computation (default 25ms)</param>
/// <param name="energy_floor">Energy floor value, used to limit the minimum energy in feature computation to avoid numerical underflow or abnormal logarithmic calculation (default 0f)</param>
/// <param name="debug_mel">Whether to enable mel-scale debugging mode. If true, additional debugging information or intermediate results will be output to verify the correctness of the mel filter bank</param>
/// <param name="window_type">Type of window function used for windowing each frame of signal (default "hamming", i.e., Hamming window; other options include ('hamming'|'hanning'|'povey'|'rectangular'|'blackman'), etc.)</param>
/// <param name="feature_type">Type of feature, specifying the type of feature to be extracted (default "fbank", i.e., filter bank feature; other options include ('fbank'|'whisper'))</param>
public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10f, float frame_length = 25f, float energy_floor = 0f, bool debug_mel = false, string window_type = "hamming", string feature_type = "fbank")
```

The following is sample code. Please configure parameters such as dither, snip_edges, sample_rate, num_bins, window_type, and feature_type according to project requirements:

```csharp
// Add project reference
using SpeechFeatures;
// Initialize OnlineFbank
OnlineFbank _onlineFbank = new OnlineFbank(
                dither: 0,
                snip_edges: false,
                sample_rate: 16000,
                num_bins: 80,
                window_type: "hamming", // window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
                feature_type: "fbank" // feature_type (string): Type of feature ('fbank'|'whisper')
                );
// Pass in audio samples to get features
public float[] GetFbank(float[] samples)
{
     float[] fbanks = _onlineFbank.GetFbank(samples);
     return fbanks;
}
```

Refer to:
----------
[1] https://github.com/manyeyes/KaldiNativeFbankSharp