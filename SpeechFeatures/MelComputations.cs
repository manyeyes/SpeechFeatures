// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.Text;

namespace SpeechFeatures
{
    public struct MelBanksOptions
    {
        public int numBins = 23;
        public float lowFreq = 20;
        public float highFreq = 0;
        public float vtlnLow = 100;
        public float vtlnHigh = -500;
        public bool debugMel = false;
        public bool htkMode = false;
        public MelBanksOptions()
        {
        }
    }

    public class MelBanks
    {
        private float[] centerFreqs;
        private (int, List<float>)[] bins;
        private bool debug;
        private bool htkMode;

        public static float InverseMelScale(float melFreq)
        {
            return 700.0f * (float)(Math.Exp(melFreq / 1127.0f) - 1.0f);
        }

        public static float MelScale(float freq)
        {
            return 1127.0f * (float)Math.Log(1.0f + freq / 700.0f);
        }

        public static float VtlnWarpFreq(float vtlnLowCutoff, float vtlnHighCutoff, float lowFreq, float highFreq, float vtlnWarpFactor, float freq)
        {
            if (freq < lowFreq || freq > highFreq)
                return freq;

            float one = 1.0f;
            float l = vtlnLowCutoff * Math.Max(one, vtlnWarpFactor);
            float h = vtlnHighCutoff * Math.Min(one, vtlnWarpFactor);
            float scale = 1.0f / vtlnWarpFactor;
            float Fl = scale * l;  // F(l);
            float Fh = scale * h;  // F(h);
            if (l > lowFreq && h < highFreq)
            {
                float scaleLeft = (Fl - lowFreq) / (l - lowFreq);
                float scaleRight = (highFreq - Fh) / (highFreq - h);

                if (freq < l)
                {
                    return lowFreq + scaleLeft * (freq - lowFreq);
                }
                else if (freq < h)
                {
                    return scale * freq;
                }
                else
                {
                    return highFreq + scaleRight * (freq - highFreq);
                }
            }
            else
            {
                return freq;
            }
        }

        public static float VtlnWarpMelFreq(float vtlnLowCutoff, float vtlnHighCutoff, float lowFreq, float highFreq, float vtlnWarpFactor, float melFreq)
        {
            return MelScale(VtlnWarpFreq(vtlnLowCutoff, vtlnHighCutoff, lowFreq, highFreq, vtlnWarpFactor, InverseMelScale(melFreq)));
        }

        public MelBanks(MelBanksOptions opts, FrameExtractionOptions frameOpts, float vtlnWarpFactor)
        {
            int numBins = opts.numBins;
            if (numBins < 3)
                throw new Exception("Must have at least 3 mel bins");

            float sampleFreq = frameOpts.SampFreq;
            int windowLengthPadded = frameOpts.PaddedWindowSize();
            if (windowLengthPadded % 2 != 0)
                throw new Exception("Window length padded must be even.");

            int numFftBins = windowLengthPadded / 2;
            float nyquist = 0.5f * sampleFreq;

            float lowFreqValue = opts.lowFreq;
            float highFreqValue;
            if (opts.highFreq > 0.0f)
                highFreqValue = opts.highFreq;
            else
                highFreqValue = nyquist + opts.highFreq;

            if (lowFreqValue < 0.0f || lowFreqValue >= nyquist || highFreqValue <= 0.0f || highFreqValue > nyquist || highFreqValue <= lowFreqValue)
                throw new Exception($"Bad values in options: low-freq {lowFreqValue} and high-freq {highFreqValue} vs. nyquist {nyquist}");

            float fftBinWidth = sampleFreq / windowLengthPadded;

            float melLowFreq = MelScale(lowFreqValue);
            float melHighFreq = MelScale(highFreqValue);

            debug = opts.debugMel;
            htkMode = opts.htkMode;

            float melFreqDelta = (melHighFreq - melLowFreq) / (numBins + 1);

            float vtlnLow = opts.vtlnLow;
            float vtlnHigh = opts.vtlnHigh;
            if (vtlnHigh < 0.0f)
                vtlnHigh += nyquist;

            if (vtlnWarpFactor != 1.0f && (vtlnLow < 0.0f || vtlnLow <= lowFreqValue || vtlnLow >= highFreqValue || vtlnHigh <= 0.0f || vtlnHigh >= highFreqValue || vtlnHigh <= vtlnLow))
                throw new Exception($"Bad values in options: vtln-low {vtlnLow} and vtln-high {vtlnHigh}, versus low-freq {lowFreqValue} and high-freq {highFreqValue}");

            bins = new (int, List<float>)[numBins];
            centerFreqs = new float[numBins];
            for (int bin = 0; bin < numBins; bin++)
            {
                float leftMel = melLowFreq + bin * melFreqDelta;
                float centerMel = melLowFreq + (bin + 1) * melFreqDelta;
                float rightMel = melLowFreq + (bin + 2) * melFreqDelta;

                if (vtlnWarpFactor != 1.0f)
                {
                    leftMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreqValue, highFreqValue, vtlnWarpFactor, leftMel);
                    centerMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreqValue, highFreqValue, vtlnWarpFactor, centerMel);
                    rightMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreqValue, highFreqValue, vtlnWarpFactor, rightMel);
                }
                centerFreqs[bin]=InverseMelScale(centerMel);

                float[] thisBin = new float[numFftBins];

                int firstIndex = -1;
                int lastIndex = -1;
                for (int i = 0; i < numFftBins; i++)
                {
                    float freq = fftBinWidth * i;
                    float mel = MelScale(freq);
                    if (mel > leftMel && mel < rightMel)
                    {
                        float weight;
                        if (mel <= centerMel)
                            weight = (mel - leftMel) / (centerMel - leftMel);
                        else
                            weight = (rightMel - mel) / (rightMel - centerMel);
                        thisBin[i]=weight;
                        if (firstIndex == -1)
                            firstIndex = i;
                        lastIndex = i;
                    }
                }
                if (firstIndex == -1 || lastIndex < firstIndex)
                    throw new Exception("You may have set num_mel_bins too large.");

                bins[bin].Item1 = firstIndex;
                if (bins[bin].Item2 == null)
                {
                    bins[bin].Item2 = new List<float>();
                }
                bins[bin].Item2.AddRange(thisBin.ToList().GetRange(firstIndex, lastIndex + 1 - firstIndex));

                if (opts.htkMode && bin == 0 && melLowFreq != 0.0f)
                {
                    bins[bin].Item2[0] = 0.0f;
                }
            }
            if (debug)
            {
                StringBuilder os = new StringBuilder();
                for (int i = 0; i < bins.Length; i++)
                {
                    os.Append($"bin {i}, offset = {bins[i].Item1}, vec = ");
                    foreach (float k in bins[i].Item2)
                    {
                        os.Append($"{k}, ");
                    }
                    os.Append("\n");
                }
                Console.WriteLine(os.ToString());
            }
        }

        public int NumBins()
        {
            return bins.Length;
        }

        public void Compute(float[] powerSpectrum, ref float[] melEnergiesOut)
        {
            int numBins = bins.Length;
            for (int i = 0; i < numBins; i++)
            {
                int offset = bins[i].Item1;
                List<float> v = bins[i].Item2;
                float energy = 0;
                for (int k = 0; k < v.Count; k++)
                {
                    energy += v[k] * powerSpectrum[k + offset];
                }
                if (htkMode && energy < 1.0)
                {
                    energy = 1.0f;
                }
                melEnergiesOut[i] = energy;
                if (float.IsNaN(energy))
                    throw new Exception($"Energy is NaN at bin {i}");
            }
            if (debug)
            {
                Console.Write("MEL BANKS:\n");
                for (int i = 0; i < numBins; i++)
                    Console.Write($"{melEnergiesOut[i]} ");
                Console.WriteLine();
            }
        }
    }
}