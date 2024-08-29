// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace SpeechFeatures
{
    public class FbankData
    {
        public float[] data;
        public int data_length;
    }

    public class FbankDatas
    {
        public float[] data;
        public int data_length;
    }

    public class KaldiFeature
    {
        private static object mutex = new object();

        public static FbankOptions GetFbankOptions(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming")
        {
            FbankOptions opts = new FbankOptions();
            opts.FrameOpts.Dither = 0;
            opts.FrameOpts.SnipEdges = snip_edges;
            opts.FrameOpts.SampFreq = sample_rate;
            opts.FrameOpts.WindowType = window_type;
            opts.FrameOpts.FrameShiftMs = frame_shift;
            opts.FrameOpts.FrameLengthMs = frame_length;
            opts.MelOpts.numBins = 80;
            opts.MelOpts.debugMel = debug_mel;
            opts.EnergyFloor = energy_floor;
            return opts;
        }

        public static OnlineFeature GetOnlineFeature(FbankOptions opts)
        {            
            OnlineFeature onlineFeature = new OnlineFeature(new FbankComputer(opts));
            return onlineFeature;
        }

        public static void AcceptWaveform(OnlineFeature onlineFeature, float sample_rate, float[] samples, int samples_size)
        {
            lock (mutex)
            {
                List<float> waveform = samples.ToList();
                onlineFeature.AcceptWaveform(sample_rate, waveform.ToArray(), waveform.Count);
            }
        }

        public static void InputFinished(OnlineFeature onlineFeature)
        {
            lock (mutex)
            {
                onlineFeature.InputFinished();
            }
        }

        public static int GetNumFramesReady(OnlineFeature onlineFeature)
        {
            lock (mutex)
            {
                return onlineFeature.NumFramesReady();
            }
        }

        public static void GetFbank(OnlineFeature onlineFeature, int currFrameIndex, ref FbankData pData)
        {
            lock (mutex)
            {
                int n = onlineFeature.NumFramesReady();
                if (n <= 0)
                {
                    throw new Exception("Please first call AcceptWaveform()");
                }
                int discard_num = currFrameIndex == 0 ? 0 : 1;
                int feature_dim = onlineFeature.Dim();
                float[] f = onlineFeature.GetFrame(currFrameIndex);
                pData.data_length = feature_dim;
                pData.data = f;
                onlineFeature.Pop(discard_num);
            }
        }

        public static void GetFbanks(OnlineFeature onlineFeature, int lastFrameIndex, ref FbankDatas pData)
        {
            lock (mutex)
            {
                int n = onlineFeature.NumFramesReady();
                if (n <= 0)
                {
                    throw new Exception("Please first call AcceptWaveform()");
                }
                List<float> features = GetFrames(onlineFeature, lastFrameIndex);
                pData.data = features.ToArray();
                pData.data_length = features.Count;
            }
        }

        private static List<float> GetFrames(OnlineFeature onlineFeature, int lastFrameIndex)
        {
            lock (mutex)
            {
                int n = onlineFeature.NumFramesReady();
                if (n - lastFrameIndex < 0)
                {
                    throw new Exception("Please first call AcceptWaveform()");
                }
                int framesNum = lastFrameIndex == 0 ? n : n - (lastFrameIndex + 1);
                int discard_num = lastFrameIndex == 0 ? n : n - (lastFrameIndex + 1);
                int feature_dim = onlineFeature.Dim();
                List<float> features = new List<float>(framesNum * feature_dim);
                int currFrameIndex = lastFrameIndex == 0 ? 0 : lastFrameIndex + 1;
                for (int i = currFrameIndex; i < n; i++)
                {
                    float[] f = onlineFeature.GetFrame(i);
                    features.AddRange(f);
                }
                onlineFeature.Pop(discard_num);
                return features;
            }
        }

    }
}