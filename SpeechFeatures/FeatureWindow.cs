// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace SpeechFeatures
{
    public struct FrameExtractionOptions
    {
        public float SampFreq { get; set; } = 16000;
        public float FrameShiftMs { get; set; } = 10.0f;   // in milliseconds.
        public float FrameLengthMs { get; set; } = 25.0f;  // in milliseconds.
        public float Dither { get; set; } = 1.0f;  // Amount of dithering, 0.0 means no dither.
        public float PreemphCoeff { get; set; } = 0.97f;  // Preemphasis coefficient.
        public bool RemoveDcOffset { get; set; } = true;  // Subtract mean of wave before FFT.
        public string WindowType { get; set; } = "povey";  // e.g. Hamming window
        // May be "hamming", "rectangular", "povey", "hanning", "sine", "blackman"
        // "povey" is a window I made to be similar to Hamming but to go to zero at
        // the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85) I just don't think the
        // Hamming window makes sense as a windowing function.
        public bool RoundToPowerOfTwo { get; set; } = true;
        public float BlackmanCoeff { get; set; } = 0.42f;
        public bool SnipEdges { get; set; } = true;
        public int MaxFeatureVectors { get; set; } = 1000;

        public FrameExtractionOptions()
        {
        }

        public int WindowShift()
        {
            return (int)(SampFreq * 0.001f * FrameShiftMs);
        }

        public int WindowSize()
        {
            return (int)(SampFreq * 0.001f * FrameLengthMs);
        }

        public int PaddedWindowSize()
        {
            return RoundToPowerOfTwo ? Utils.FeatureFunctions.RoundUpToNearestPowerOfTwo(WindowSize()) : WindowSize();
        }
    }

    public class FeatureWindowFunction
    {
        private List<float> window;

        public FeatureWindowFunction(FrameExtractionOptions opts)
        {
            int frameLength = opts.WindowSize();
            if (frameLength <= 0)
            {
                throw new ArgumentException("Frame length must be greater than 0.");
            }
            window = new List<float>(frameLength);

            double a = 2 * Math.PI / (frameLength - 1);
            for (int i = 0; i < frameLength; i++)
            {
                double iFl = i;
                switch (opts.WindowType)
                {
                    case "hanning":
                        window.Add(0.5f - 0.5f * (float)Math.Cos(a * iFl));
                        break;
                    case "sine":
                        window.Add((float)Math.Sin(0.5f * a * iFl));
                        break;
                    case "hamming":
                        window.Add(0.54f - 0.46f * (float)Math.Cos(a * iFl));
                        break;
                    case "povey":
                        window.Add((float)Math.Pow(0.5 - 0.5 * (float)Math.Cos(a * iFl), 0.85));
                        break;
                    case "rectangular":
                        window.Add(1.0f);
                        break;
                    case "blackman":
                        window.Add(opts.BlackmanCoeff - 0.5f * (float)Math.Cos(a * iFl) + (0.5f - opts.BlackmanCoeff) * (float)Math.Cos(2 * a * iFl));
                        break;
                    default:
                        throw new ArgumentException($"Invalid window type {opts.WindowType}");
                }
            }
        }

        public void Apply(float[] wave)
        {
            for (int k = 0; k < window.Count; k++)
            {
                wave[k] *= window[k];
            }
        }
    }

    public class FeatureWindow
    {
        public static long FirstSampleOfFrame(int frame, FrameExtractionOptions opts)
        {
            int frameShift = opts.WindowShift();
            if (opts.SnipEdges)
            {
                return frame * frameShift;
            }
            else
            {
                long midpointOfFrame = frameShift * frame + frameShift / 2;
                long beginningOfFrame = midpointOfFrame - opts.WindowSize() / 2;
                return beginningOfFrame;
            }
        }

        public static int NumFrames(long numSamples, FrameExtractionOptions opts, bool flush = true)
        {
            int frameShift = opts.WindowShift();
            int frameLength = opts.WindowSize();
            if (opts.SnipEdges)
            {
                if (numSamples < frameLength)
                {
                    return 0;
                }
                else
                {
                    return 1 + (int)((numSamples - frameLength) / frameShift);
                }
            }
            else
            {
                int numFrames = (int)(numSamples + (frameShift / 2)) / frameShift;
                if (flush)
                {
                    return numFrames;
                }
                else
                {
                    long endSampleOfLastFrame = FirstSampleOfFrame(numFrames - 1, opts) + frameLength;
                    while (numFrames > 0 && endSampleOfLastFrame > numSamples)
                    {
                        numFrames--;
                        endSampleOfLastFrame -= frameShift;
                    }
                    return numFrames;
                }
            }
        }

        public static void ExtractWindow(long sampleOffset, List<float> wave, int f, FrameExtractionOptions opts, FeatureWindowFunction windowFunction, List<float> window, ref float? logEnergyPreWindow)
        {
            if (sampleOffset < 0 || wave.Count == 0)
            {
                throw new ArgumentException("Invalid arguments.");
            }

            int frameLength = opts.WindowSize();
            int frameLengthPadded = opts.PaddedWindowSize();

            long numSamples = sampleOffset + wave.Count;
            long startSample = FirstSampleOfFrame(f, opts);
            long endSample = startSample + frameLength;

            if (opts.SnipEdges)
            {
                if (startSample < sampleOffset || endSample > numSamples)
                {
                    throw new ArgumentException("Invalid range.");
                }
            }
            else
            {
                if (sampleOffset != 0 && startSample < sampleOffset)
                {
                    throw new ArgumentException("Invalid start sample.");
                }
            }
            if (window.Count != frameLengthPadded)
            {
                window.Clear();
                window.Capacity = frameLengthPadded;
                for (int i = 0; i < frameLengthPadded; i++)
                {
                    window.Add(0f);
                }
            }
            int waveStart = (int)(startSample - sampleOffset);
            int waveEnd = waveStart + frameLength;
            if (waveStart >= 0 && waveEnd <= wave.Count)
            {
                for (int s = 0; s < frameLength; s++)
                {
                    window[s] = wave[waveStart + s];
                }
            }
            else
            {
                int waveDim = wave.Count;
                for (int s = 0; s < frameLength; s++)
                {
                    int sInWave = s + waveStart;
                    while (sInWave < 0 || sInWave >= waveDim)
                    {
                        if (sInWave < 0)
                        {
                            sInWave = -sInWave - 1;
                        }
                        else
                        {
                            sInWave = 2 * waveDim - 1 - sInWave;
                        }
                    }
                    window[s] = wave[sInWave];
                }
            }
            ProcessWindow(opts, windowFunction, window.ToArray(), ref logEnergyPreWindow);
        }

        public static void ProcessWindow(FrameExtractionOptions opts, FeatureWindowFunction windowFunction, float[] window, ref float? logEnergyPreWindow)
        {
            int frameLength = opts.WindowSize();
            if (opts.RemoveDcOffset)
            {
                Utils.FeatureFunctions.RemoveDcOffset(window, frameLength);
            }
            if (logEnergyPreWindow != null)
            {
                float energy = Math.Max(Utils.FeatureFunctions.InnerProduct(window, window, frameLength), float.Epsilon);
                logEnergyPreWindow = (float)Math.Log(energy);
            }
            if (opts.PreemphCoeff != 0.0f)
            {
                Utils.FeatureFunctions.Preemphasize(window, frameLength, opts.PreemphCoeff);
            }
            windowFunction.Apply(window);
        }
    }
}