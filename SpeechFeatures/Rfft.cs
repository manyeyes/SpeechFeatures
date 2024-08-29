// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using SpeechFeatures.Utils;

namespace SpeechFeatures
{

    public class Rfft
    {
        private RfftImpl impl;

        public Rfft(int n)
        {
            impl = new RfftImpl(n);
        }

        ~Rfft()
        {
        }

        public void Compute(List<float> inOut)
        {
            List<double> d = inOut.ConvertAll(f => (double)f);
            double[] dTemp = d.ToArray();
            impl.Compute(ref dTemp);
            d=dTemp.ToList();
            for (int i = 0; i < inOut.Count; i++)
            {
                inOut[i] = (float)d[i];
            }
        }

        public void Compute(ref double[] inOut)
        {
            impl.Compute(ref inOut);
        }
    }

    public class RfftImpl
    {
        private int n;
        private int[] ip;
        private double[] w;

        public RfftImpl(int n)
        {
            this.n = n;
            // Check if n is a power of 2.
            if ((n & (n - 1)) != 0)
                throw new Exception("n must be a power of 2.");

            int ipSize = 2 + (int)Math.Sqrt(n / 2);
            ip = new int[ipSize];
            w = new double[n / 2];
        }

        public void Compute(ref float[] inOut)
        {
            List<double> d = inOut.ToList().ConvertAll(f => (double)f);
            double[] dTemp = d.ToArray();
            Compute(ref dTemp);
            d = dTemp.ToList();
            for (int i = 0; i < inOut.Length; i++)
            {
                inOut[i] = (float)d[i];
            }
        }

        /// <summary>
        /// Forward FFT sign is 1.
        /// </summary>
        /// <param name="inOut"></param>
        public unsafe void Compute(ref double[] inOut)
        {
            // 1 means forward fft
            fixed (double* ptrInOut = inOut)
            fixed (int* ptrIp = ip)
            fixed (double* ptrW = w)
            {
                Fftsg.rdft(n, 1, ptrInOut, ptrIp, ptrW);
            }
        }
    }
}