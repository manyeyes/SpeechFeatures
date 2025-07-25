// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
//Due to the fact that the zipformer 2ctc xlarge model is trained using similar Whisper features, in order to match the feature rules of this type of model, we refer to the Kaldi native fbank project to implement the same LibrosaMelBanks calculation method.
//This implementation is copied from kaldi-native-fbank/csrc/whisper-feature.cc
using System.Text;

namespace SpeechFeatures
{
    public class WhisperComputer : IFeatureComputer
    {
        public FeatureOptions _opts { get; }
        private Dictionary<float, MelBanks2> _melBanks=new Dictionary<float, MelBanks2>();
        // 数学常数：2π
        public const double M_2PI = 6.283185307179586476925286766559005;

        public WhisperComputer(FeatureOptions opts)
        {
            opts.FrameOpts.Dither = 0;
            opts.FrameOpts.SnipEdges = false;
            opts.FrameOpts.SampFreq = 16000;
            opts.FrameOpts.WindowType = "hanning";
            opts.FrameOpts.FrameShiftMs = 10;
            opts.FrameOpts.FrameLengthMs = 25;
            opts.FrameOpts.RoundToPowerOfTwo = false;
            opts.FrameOpts.RemoveDcOffset = false;
            opts.FrameOpts.PreemphCoeff = 0;
            //opts.MelOpts.numBins = 80;
            opts.MelOpts.lowFreq = 0;
            //opts.MelOpts.highFreq = -400;
            opts.MelOpts.isLibrosa = true;
            //opts.MelOpts.debugMel = debug_mel;
            //opts.EnergyFloor = energy_floor;
            _opts = opts;
            // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
            GetMelBanks(1.0f);
        }

        ~WhisperComputer()
        {
            foreach (var pair in _melBanks)
            {
                //pair.Value.Dispose();
                _melBanks.Remove(pair.Key);
            }
        }

        /// <summary>
        /// 离散傅里叶变换(DFT)实现
        /// </summary>
        /// <param name="in">输入实数序列</param>
        /// <returns>输出复数序列，以float[2]形式存储，每个元素为(re, im)</returns>
        public static float[] Dft(List<float> @in)
        {
            int N = @in.Count;
            float[] output = new float[N * 2];  // 每个复数占两个元素位置

            double M_2PI_over_N = M_2PI / N;

            for (int k = 0; k < N; k++)
            {
                float re = 0;
                float im = 0;

                for (int n = 0; n < N; n++)
                {
                    double angle = M_2PI_over_N * k * n;
                    re += @in[n] * (float)Math.Cos(angle);
                    im -= @in[n] * (float)Math.Sin(angle);
                }

                output[k * 2] = re;     // 实部
                output[k * 2 + 1] = im; // 虚部
            }

            return output;
        }

        /// <summary>
        /// Cooley-Tukey快速傅里叶变换(FFT)实现
        /// </summary>
        /// <param name="in">输入实数序列</param>
        /// <returns>输出复数序列，以float[2]形式存储，每个元素为(re, im)</returns>
        public static float[] Fft(List<float> @in)
        {
            int N = @in.Count;
            float[] output = new float[N * 2];

            // 基本情况：序列长度为1
            if (N == 1)
            {
                output[0] = @in[0];
                output[1] = 0;
                return output;
            }

            // 若序列长度为奇数，使用DFT
            if (N % 2 == 1)
            {
                return Dft(@in);
            }

            // 分离偶数索引和奇数索引的元素
            List<float> even = new List<float>(N / 2);
            List<float> odd = new List<float>(N / 2);

            for (int i = 0; i < N; i++)
            {
                if (i % 2 == 0)
                    even.Add(@in[i]);
                else
                    odd.Add(@in[i]);
            }

            // 递归计算偶数和奇数序列的FFT
            float[] evenFft = Fft(even);
            float[] oddFft = Fft(odd);

            // 合并结果
            for (int k = 0; k < N / 2; k++)
            {
                double theta = M_2PI * k / N;
                float re = (float)Math.Cos(theta);  // 旋转因子的实部
                float im = (float)-Math.Sin(theta); // 旋转因子的虚部

                // 奇数序列FFT结果
                float reOdd = oddFft[2 * k];
                float imOdd = oddFft[2 * k + 1];

                // 计算X(k) = E(k) + W_N^k * O(k)
                output[2 * k] = evenFft[2 * k] + re * reOdd - im * imOdd;
                output[2 * k + 1] = evenFft[2 * k + 1] + re * imOdd + im * reOdd;

                // 计算X(k + N/2) = E(k) - W_N^k * O(k)
                output[2 * (k + N / 2)] = evenFft[2 * k] - re * reOdd + im * imOdd;
                output[2 * (k + N / 2) + 1] = evenFft[2 * k + 1] - re * imOdd - im * reOdd;
            }

            return output;
        }

        private MelBanks2 GetMelBanks(float vtlnWarp)
        {
            MelBanks2? thisMelBanks = null;
            if (!_melBanks.TryGetValue(vtlnWarp, out thisMelBanks))
            {
                thisMelBanks = new MelBanks2(_opts.MelOpts, _opts.FrameOpts, vtlnWarp);
                _melBanks[vtlnWarp] = thisMelBanks;
            }
            return thisMelBanks;
        }

        public int Dim()
        {
            return _opts.MelOpts.numBins + (_opts.UseEnergy ? 1 : 0);
        }

        public bool NeedRawLogEnergy()
        {
            return _opts.UseEnergy && _opts.RawEnergy;
        }
        public void Compute(float? signalRawLogEnergy, float vtlnWarp, List<float> signalFrame,ref float[] feature)
        {
            // 验证输入帧大小是否符合预期
            int paddedWindowSize = _opts.FrameOpts.PaddedWindowSize();
            if (signalFrame.Count != paddedWindowSize)
                throw new ArgumentException($"信号帧大小必须为{paddedWindowSize}", nameof(signalFrame));

            // 计算FFT
            float[] fftOut = Fft(signalFrame);

            // 计算功率谱
            int numFft = signalFrame.Count;
            int powerSize = numFft / 2 + 1;
            float[] power = new float[powerSize];

            for (int i = 0; i < powerSize; i++)
            {
                float re = fftOut[2 * i];       // 实部
                float im = fftOut[2 * i + 1];   // 虚部
                power[i] = re * re + im * im;   // 计算功率（模的平方）
            }

            var melBanks = GetMelBanks(vtlnWarp);
            // 验证输出特征数组大小
            if (feature.Length < melBanks.NumBins)
                throw new ArgumentException($"输出特征数组大小必须至少为{melBanks.NumBins}", nameof(feature));

            // 使用梅尔滤波器组计算梅尔特征
            int melOffset = _opts.UseEnergy && !_opts.HtkCompat ? 1 : 0;
            float[] melEnergies = new float[feature.Length - melOffset];
            // Sum with mel filter banks over the power spectrum
            melBanks.Compute(power, ref melEnergies);
            Array.Copy(melEnergies, 0, feature, melOffset, melEnergies.Length);
        }

        public FrameExtractionOptions GetFrameOptions()
        {
            return _opts.FrameOpts;
        }
    }


}