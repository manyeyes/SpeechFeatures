// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace SpeechFeatures.Utils
{
    internal class FeatureFunctions
    {
        public static int RoundUpToNearestPowerOfTwo(int n)
        {
            // copied from kaldi/src/base/kaldi-math.cc
            if (n <= 0)
            {
                throw new ArgumentException("n must be greater than 0.");
            }
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return n + 1;
        }

        public static void ComputePowerSpectrum(List<float> complexFft)
        {
            int dim = complexFft.Count;

            // now we have in complex_fft, first half of complex spectrum
            // it's stored as [real0, realN/2, real1, im1, real2, im2,...]

            float[] p = complexFft.ToArray();
            int halfDim = dim / 2;
            float firstEnergy = p[0] * p[0];
            float lastEnergy = p[1] * p[1];

            for (int i = 1; i < halfDim; i++)
            {
                float real = p[i * 2];
                float im = p[i * 2 + 1];
                p[i] = real * real + im * im;
            }
            p[0] = firstEnergy;
            p[halfDim] = lastEnergy;
            complexFft.Clear();
            complexFft.AddRange(p);
        }

        public static void RemoveDcOffset(float[] d, int n)
        {
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += d[i];
            }

            float mean = sum / n;

            for (int i = 0; i < n; i++)
            {
                d[i] -= mean;
            }
        }
        // Implementations of FrameExtractionOptions, MelBanksOptions, MelBanks, and Rfft classes would be needed.
        public static float InnerProduct(float[] a, float[] b, int n)
        {
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        public static void Preemphasize(float[] d, int n, float preemphCoeff)
        {
            if (preemphCoeff == 0.0f)
            {
                return;
            }

            if (preemphCoeff < 0.0f || preemphCoeff > 1.0f)
            {
                throw new ArgumentException("Preemphasis coefficient must be between 0 and 1.");
            }

            for (int i = n - 1; i > 0; i--)
            {
                d[i] -= preemphCoeff * d[i - 1];
            }
            d[0] -= preemphCoeff * d[0];
        }
    }
}
