// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
//Due to the fact that the zipformer 2ctc xlarge model is trained using similar Whisper features, in order to match the feature rules of this type of model, we refer to the Kaldi native fbank project to implement the same LibrosaMelBanks calculation method.
//This implementation is copied from kaldi-native-fbank/csrc/mel-computations.cc
using System.Text;

namespace SpeechFeatures
{
    public struct MelBanksOptions
    {
        public int numBins = 25;
        public float lowFreq = 20f;
        public float highFreq = 0f;
        public float vtlnLow = 100f;
        public float vtlnHigh = -500f;
        public bool debugMel = false;
        public bool htkMode = false;
        public bool isLibrosa = false;
        public string norm = "slaney";
        public bool useSlaneyMelScale = true;
        public bool floorToIntBin = false;
        public MelBanksOptions()
        {
        }

        public override string ToString()
        {
            return $"num_bins: {numBins}\nlow_freq: {lowFreq}\nhigh_freq: {highFreq}\nvtln_low: {vtlnLow}\nvtln_high: {vtlnHigh}\ndebug_mel: {debugMel}\nhtk_mode: {htkMode}\n";
        }
    }
    public class BinItem
    {
        public int Item1 { get; set; }
        public List<float> Item2 { get; set; }
        public BinItem(int key, List<float> values)
        {
            Item1 = key;
            Item2 = values;
        }
    }
    /// <summary>
    /// 梅尔滤波器组实现
    /// </summary>
    public class MelBanks2
    {
#if NET471_OR_GREATER || NET6_0_OR_GREATER
        // 存储每个梅尔 bin 的起始索引和权重（对应C++的std::vector<std::pair<int32_t, std::vector<float>>>）
        private List<Tuple<int, List<float>>> _bins = new List<Tuple<int, List<float>>>();
#else
        private List<BinItem> _bins = new List<BinItem>();
#endif
        private bool _debug;
        private bool _htkMode;

        // 梅尔频率与线性频率转换（HTK模式）
        public static float MelScale(float freq) => 1127.0f * (float)Math.Log(1.0f + freq / 700.0f);
        public static float InverseMelScale(float melFreq) => 700.0f * (float)(Math.Exp(melFreq / 1127.0f) - 1.0f);

        // 梅尔频率与线性频率转换（Slaney模式，用于librosa兼容）
        public static float MelScaleSlaney(float freq)
        {
            if (freq <= 1000)
                return freq * 3 / 200.0f;
            // return 15 + 27 * logf(freq / 1000) / logf(6.4f)
            // Note: 27/log(6.4) = 14.545078505785561
            return 15 + 14.545078505785561f * (float)Math.Log(freq / 1000);
        }

        public static float InverseMelScaleSlaney(float melFreq)
        {
            if (melFreq <= 15)
                return 200.0f / 3 * melFreq;
            // return 1000 * expf((mel_freq - 15) * logf(6.4f) / 27);
            // Note: log(6.4)/27 = 0.06875177742094911
            return 1000 * (float)Math.Exp((melFreq - 15) * 0.06875177742094911f);
        }

        // VTLN频率扭曲
        public static float VtlnWarpFreq(float vtlnLowCutoff, float vtlnHighCutoff,
                                        float lowFreq, float highFreq,
                                        float vtlnWarpFactor, float freq)
        {
            if (freq < lowFreq || freq > highFreq)
                return freq;

            if (vtlnLowCutoff <= lowFreq || vtlnHighCutoff >= highFreq || vtlnHighCutoff <= vtlnLowCutoff)
                throw new ArgumentException($"Invalid VTLN parameters: vtlnLow={vtlnLowCutoff}, vtlnHigh={vtlnHighCutoff}, lowFreq={lowFreq}, highFreq={highFreq}");

            float one = 1.0f;
            float l = vtlnLowCutoff * Math.Max(one, vtlnWarpFactor);
            float h = vtlnHighCutoff * Math.Min(one, vtlnWarpFactor);
            float scale = 1.0f / vtlnWarpFactor;
            float fl = scale * l;
            float fh = scale * h;

            if (l <= lowFreq || h >= highFreq)
                throw new InvalidOperationException("VTLN warp points out of valid range");

            float scaleLeft = (fl - lowFreq) / (l - lowFreq);
            float scaleRight = (highFreq - fh) / (highFreq - h);

            if (freq < l)
                return lowFreq + scaleLeft * (freq - lowFreq);
            if (freq < h)
                return scale * freq;
            return highFreq + scaleRight * (freq - highFreq);
        }

        // 基于梅尔频率的VTLN扭曲
        public static float VtlnWarpMelFreq(float vtlnLowCutoff, float vtlnHighCutoff,
                                           float lowFreq, float highFreq,
                                           float vtlnWarpFactor, float melFreq)
        {
            return MelScale(VtlnWarpFreq(vtlnLowCutoff, vtlnHighCutoff, lowFreq, highFreq, vtlnWarpFactor, InverseMelScale(melFreq)));
        }

        // 构造函数
        public MelBanks2(MelBanksOptions opts, FrameExtractionOptions frameOpts, float vtlnWarpFactor = 1.0f)
        {
            _debug = opts.debugMel;
            _htkMode = opts.htkMode;

            if (opts.isLibrosa)
                InitLibrosaMelBanks(opts, frameOpts, vtlnWarpFactor);
            else
                InitKaldiMelBanks(opts, frameOpts, vtlnWarpFactor);
        }

        // 从权重矩阵初始化
        public MelBanks2(float[] weights, int numRows, int numCols)
        {
            _debug = false;
            _htkMode = false;
#if NET471_OR_GREATER || NET6_0_OR_GREATER
            _bins = new List<Tuple<int, List<float>>>();
#else
            _bins = new List<BinItem>();
#endif
            for (int bin = 0; bin < numRows; bin++)
            {
                int startIdx = bin * numCols;
                int firstIndex = -1;
                int lastIndex = -1;

                // 找到非零权重的范围
                for (int i = 0; i < numCols; i++)
                {
                    if (weights[startIdx + i] != 0)
                    {
                        if (firstIndex == -1)
                            firstIndex = i;
                        lastIndex = i;
                    }
                }

                if (firstIndex == -1 || lastIndex < firstIndex)
                    throw new ArgumentException("Invalid weight matrix: empty or invalid bin");

                // 提取权重
                List<float> binWeights = new List<float>();
                for (int i = firstIndex; i <= lastIndex; i++)
                    binWeights.Add(weights[startIdx + i]);
#if NET471_OR_GREATER || NET6_0_OR_GREATER
                _bins.Add(Tuple.Create(firstIndex, binWeights));
#else
                _bins.Add(new BinItem(firstIndex, binWeights));
#endif
            }
        }

        // 初始化Kaldi风格的梅尔滤波器组
        private void InitKaldiMelBanks(MelBanksOptions opts, FrameExtractionOptions frameOpts, float vtlnWarpFactor)
        {
            int numBins = opts.numBins;
            if (numBins < 3)
                throw new ArgumentException("Must have at least 3 mel bins", nameof(opts));

            float sampleFreq = frameOpts.SampFreq;
            int windowLengthPadded = frameOpts.PaddedWindowSize();
            if (windowLengthPadded % 2 != 0)
                throw new InvalidOperationException("Window length must be even");

            int numFftBins = windowLengthPadded / 2;
            float nyquist = 0.5f * sampleFreq;

            // 处理高低频截止
            float lowFreq = opts.lowFreq;
            float highFreq = opts.highFreq > 0 ? opts.highFreq : nyquist + opts.highFreq;

            if (lowFreq < 0 || lowFreq >= nyquist || highFreq <= 0 || highFreq > nyquist || highFreq <= lowFreq)
                throw new ArgumentException($"Invalid frequency range: lowFreq={lowFreq}, highFreq={highFreq}, nyquist={nyquist}");

            float fftBinWidth = sampleFreq / windowLengthPadded;
            float melLow = MelScale(lowFreq);
            float melHigh = MelScale(highFreq);
            float melDelta = (melHigh - melLow) / (numBins + 1);

            // 处理VTLN参数
            float vtlnLow = opts.vtlnLow;
            float vtlnHigh = opts.vtlnHigh < 0 ? nyquist + opts.vtlnHigh : opts.vtlnHigh;

            if (vtlnWarpFactor != 1.0f && (vtlnLow <= lowFreq || vtlnHigh >= highFreq || vtlnHigh <= vtlnLow))
                throw new ArgumentException($"Invalid VTLN parameters for warping", nameof(opts));

            // 计算每个梅尔bin的权重
#if NET471_OR_GREATER || NET6_0_OR_GREATER
            _bins = new List<Tuple<int, List<float>>>();
#else
            _bins = new List<BinItem>();
#endif
            for (int bin = 0; bin < numBins; bin++)
            {
                // 计算梅尔频率范围
                float leftMel = melLow + bin * melDelta;
                float centerMel = melLow + (bin + 1) * melDelta;
                float rightMel = melLow + (bin + 2) * melDelta;

                // 应用VTLN扭曲
                if (vtlnWarpFactor != 1.0f)
                {
                    leftMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, leftMel);
                    centerMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, centerMel);
                    rightMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, rightMel);
                }

                // 计算FFT bin权重
                List<float> weights = new List<float>();
                int firstIndex = -1;
                int lastIndex = -1;

                for (int i = 0; i < numFftBins; i++)
                {
                    float freq = fftBinWidth * i;
                    float mel = MelScale(freq);

                    if (mel > leftMel && mel < rightMel)
                    {
                        float weight = mel <= centerMel
                            ? (mel - leftMel) / (centerMel - leftMel)
                            : (rightMel - mel) / (rightMel - centerMel);

                        if (firstIndex == -1)
                            firstIndex = i;
                        lastIndex = i;
                        weights.Add(weight);
                    }
                }

                if (firstIndex == -1 || lastIndex < firstIndex)
                    throw new InvalidOperationException("Insufficient FFT bins for mel bin (num_bins may be too large)");

                // HTK兼容的特殊处理
                if (opts.htkMode && bin == 0 && melLow != 0)
                    weights[0] = 0.0f;

#if NET471_OR_GREATER || NET6_0_OR_GREATER
                _bins.Add(Tuple.Create(firstIndex, weights));
#else
                _bins.Add(new BinItem(firstIndex, weights));
#endif
            }

            // 调试输出
            if (_debug)
                DebugOutputBins();
        }

        // 初始化librosa风格的梅尔滤波器组
        private void InitLibrosaMelBanks(MelBanksOptions opts, FrameExtractionOptions frameOpts, float vtlnWarpFactor)
        {
            int numBins = opts.numBins;
            if (numBins < 3)
                throw new ArgumentException("Must have at least 3 mel bins", nameof(opts));

            float sampleFreq = frameOpts.SampFreq;
            int windowLengthPadded = frameOpts.PaddedWindowSize();
            if (windowLengthPadded % 2 != 0)
                throw new InvalidOperationException("Window length must be even");

            int numFftBins = windowLengthPadded / 2;
            float nyquist = 0.5f * sampleFreq;

            // 处理高低频截止
            float lowFreq = opts.lowFreq;
            float highFreq = opts.highFreq > 0 ? opts.highFreq : nyquist + opts.highFreq;

            if (lowFreq < 0 || lowFreq >= nyquist || highFreq <= 0 || highFreq > nyquist || highFreq <= lowFreq)
                throw new ArgumentException($"Invalid frequency range: lowFreq={lowFreq}, highFreq={highFreq}, nyquist={nyquist}");

            float fftBinWidth = sampleFreq / windowLengthPadded;
            bool useSlaney = opts.useSlaneyMelScale;
            bool slaneyNorm = opts.norm == "slaney";

            // 计算梅尔频率范围
            float melLow = useSlaney ? MelScaleSlaney(lowFreq) : MelScale(lowFreq);
            float melHigh = useSlaney ? MelScaleSlaney(highFreq) : MelScale(highFreq);
            float melDelta = (melHigh - melLow) / (numBins + 1);

#if NET471_OR_GREATER || NET6_0_OR_GREATER
            _bins = new List<Tuple<int, List<float>>>();
#else
            _bins = new List<BinItem>();
#endif
            for (int bin = 0; bin < numBins; bin++)
            {
                // 计算梅尔频率范围
                float leftMel = melLow + bin * melDelta;
                float centerMel = melLow + (bin + 1) * melDelta;
                float rightMel = melLow + (bin + 2) * melDelta;

                // 转换回线性频率
                float leftHz = useSlaney ? InverseMelScaleSlaney(leftMel) : InverseMelScale(leftMel);
                float centerHz = useSlaney ? InverseMelScaleSlaney(centerMel) : InverseMelScale(centerMel);
                float rightHz = useSlaney ? InverseMelScaleSlaney(rightMel) : InverseMelScale(rightMel);

                // 处理整数bin修正
                if (opts.floorToIntBin)
                {
                    float scale = (windowLengthPadded + 1.0f) / sampleFreq;
                    leftHz = (int)(leftHz * scale);
                    centerHz = (int)(centerHz * scale);
                    rightHz = (int)(rightHz * scale);
                }

                // 计算FFT bin权重
                List<float> weights = new List<float>();
                int firstIndex = -1;
                int lastIndex = -1;
                int fftRange = opts.floorToIntBin ? numFftBins + 1 : numFftBins;

                for (int i = 0; i < fftRange; i++)
                {
                    float hz = opts.floorToIntBin ? i : fftBinWidth * i;

                    if (hz > leftHz && hz < rightHz)
                    {
                        float weight = hz <= centerHz
                            ? (hz - leftHz) / (centerHz - leftHz)
                            : (rightHz - hz) / (rightHz - centerHz);

                        // Slaney归一化
                        if (slaneyNorm)
                            weight *= 2 / (rightHz - leftHz);

                        if (firstIndex == -1)
                            firstIndex = i;
                        lastIndex = i;
                        weights.Add(weight);
                    }
                }

                if (firstIndex == -1 || lastIndex < firstIndex)
                    throw new InvalidOperationException("Insufficient FFT bins for mel bin (num_bins may be too large)");

#if NET471_OR_GREATER || NET6_0_OR_GREATER
                _bins.Add(Tuple.Create(firstIndex, weights));
#else
                _bins.Add(new BinItem(firstIndex, weights));
#endif
            }

            // 调试输出
            if (_debug)
                DebugOutputBins();
        }

        // 计算梅尔能量
        public void Compute(float[] powerSpectrum, ref float[] melEnergiesOut)
        {
            if (powerSpectrum == null)
                throw new ArgumentNullException(nameof(powerSpectrum));
            if (melEnergiesOut == null)
                throw new ArgumentNullException(nameof(melEnergiesOut));
            if (melEnergiesOut.Length < _bins.Count)
                throw new ArgumentException("Output array too small", nameof(melEnergiesOut));

            for (int i = 0; i < _bins.Count; i++)
            {
#if NET471_OR_GREATER || NET6_0_OR_GREATER
                var (offset, weights) = _bins[i];
#else
                var offset = _bins[i].Item1;
                var weights = _bins[i].Item2;
#endif
                float energy = 0;

                for (int k = 0; k < weights.Count; k++)
                {
                    int idx = offset + k;
                    if (idx >= powerSpectrum.Length)
                        throw new IndexOutOfRangeException("Power spectrum array too small for mel bin weights");
                    energy += weights[k] * powerSpectrum[idx];
                }

                // HTK风格的能量下限
                if (_htkMode && energy < 1.0f)
                    energy = 1.0f;

                melEnergiesOut[i] = energy;

                // 检查NaN
                if (float.IsNaN(energy))
                    throw new InvalidOperationException("Computed NaN energy in mel bank");
            }

            // 调试输出
            if (_debug)
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("Mel energies: ");
                foreach (var e in melEnergiesOut.Take(_bins.Count))
                    sb.Append($"{e:F2} ");
                System.Diagnostics.Debug.WriteLine(sb.ToString());
            }
        }

        // 获取梅尔bin数量
        public int NumBins => _bins.Count;

        // 调试输出滤波器组信息
        private void DebugOutputBins()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < _bins.Count; i++)
            {
#if NET471_OR_GREATER || NET6_0_OR_GREATER
                var (offset, weights) = _bins[i];
#else
                var offset = _bins[i].Item1;
                var weights = _bins[i].Item2;
#endif
                sb.AppendLine($"Bin {i}, Offset: {offset}, Weights: {string.Join(", ", weights.Select(w => w.ToString("F4")))}");
            }
            System.Diagnostics.Debug.WriteLine(sb.ToString());
        }
    }

    /// <summary>
    /// 计算提升系数（用于倒谱系数）
    /// </summary>
    public static class LifterCoeffs
    {
        public static void Compute(float q, List<float> coeffs)
        {
            if (coeffs == null)
                throw new ArgumentNullException(nameof(coeffs));

            for (int i = 0; i < coeffs.Count; i++)
                coeffs[i] = 1.0f + 0.5f * q * (float)Math.Sin(Math.PI * i / q);
        }
    }

}