// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace SpeechFeatures
{
    class RecyclingVector
    {
        private List<List<float>> items;
        private int itemsToHold;
        private int firstAvailableIndex;

        public RecyclingVector(int itemsToHold = -1)
        {
            this.itemsToHold = itemsToHold == 0 ? -1 : itemsToHold;
            firstAvailableIndex = 0;
            items = new List<List<float>>();
        }

        public float[] At(int index)
        {
            if (index < firstAvailableIndex)
                throw new Exception($"Attempted to retrieve feature vector that was already removed by the RecyclingVector (index = {index}; first_available_index = {firstAvailableIndex}; size = {Size()})");
            return items[index - firstAvailableIndex].ToArray();
        }

        public void PushBack(List<float> item)
        {
            if (items.Count == itemsToHold)
            {
                items.RemoveAt(0);
                firstAvailableIndex++;
            }
            items.Add(item);
        }

        public int Size()
        {
            return firstAvailableIndex + items.Count;
        }

        public void Pop(int n)
        {
            for (int i = 0; i < n && items.Count > 0; i++)
            {
                items.RemoveAt(0);
                firstAvailableIndex++;
            }
        }
    }

    public class OnlineGenericBaseFeature<T> where T : IFeatureComputer
    {
        private T computer;
        private FeatureWindowFunction windowFunction;
        private RecyclingVector features;
        private bool inputFinished;
        private long waveformOffset;
        private List<float> waveformRemainder;

        public OnlineGenericBaseFeature(T opts)
        {
            computer = opts;
            windowFunction = new FeatureWindowFunction(computer.GetFrameOptions());
            inputFinished = false;
            waveformOffset = 0;
            features = new RecyclingVector();
            waveformRemainder = new List<float>();

            if (computer.GetFrameOptions().MaxFeatureVectors <= 200)
                throw new Exception("Online feature extraction requires more than 200 max feature vectors.");
        }

        public int Dim()
        {
            return computer.Dim();
        }

        public float FrameShiftInSeconds()
        {
            return computer.GetFrameOptions().FrameShiftMs / 1000.0f;
        }

        public int NumFramesReady()
        {
            return features.Size();
        }

        public bool IsLastFrame(int frame)
        {
            return inputFinished && frame == NumFramesReady() - 1;
        }

        public float[] GetFrame(int frame)
        {
            return features.At(frame);
        }

        public void AcceptWaveform(float samplingRate, float[] waveform, int n)
        {
            if (n == 0)
                return;

            if (inputFinished)
                throw new Exception("AcceptWaveform called after InputFinished() was called.");

            if (samplingRate != computer.GetFrameOptions().SampFreq)
                throw new Exception("Sampling rate mismatch.");

            waveformRemainder.AddRange(waveform);
            ComputeFeatures();
        }

        public void InputFinished()
        {
            inputFinished = true;
            ComputeFeatures();
        }

        public void Pop(int n)
        {
            features.Pop(n);
        }

        private void ComputeFeatures()
        {
            var frameOpts = computer.GetFrameOptions();
            long numSamplesTotal = waveformOffset + waveformRemainder.Count;
            int numFramesOld = features.Size();
            int numFramesNew = FeatureWindow.NumFrames(numSamplesTotal, frameOpts, inputFinished);
            if (numFramesNew < numFramesOld)
                throw new Exception("Number of new frames cannot be less than old frames.");
            // Note: this online feature-extraction code does not support VTLN.
            float vtlnWarp = 1.0f;
            List<float> window = new List<float>();
            bool needRawLogEnergy = computer.NeedRawLogEnergy();
            for (int frame = numFramesOld; frame < numFramesNew; frame++)
            {
                window.Clear();
                float? rawLogEnergy = 0.0f;
                float? rawLogEnergyRef = null;
                if (needRawLogEnergy)
                {
                    FeatureWindow.ExtractWindow(waveformOffset, waveformRemainder, frame, frameOpts, windowFunction, window, ref rawLogEnergy);
                }
                else
                {
                    FeatureWindow.ExtractWindow(waveformOffset, waveformRemainder, frame, frameOpts, windowFunction, window, ref rawLogEnergyRef);
                }
                float[] thisFeature = new float[computer.Dim()];
                computer.Compute(rawLogEnergy, vtlnWarp, window, ref thisFeature);
                features.PushBack(thisFeature.ToList());
            }
            // OK, we will now discard any portion of the signal that will not be
            // necessary to compute frames in the future.
            long firstSampleOfNextFrame = FirstSampleOfFrame(numFramesNew, frameOpts);
            int samplesToDiscard = (int)(firstSampleOfNextFrame - waveformOffset);
            if (samplesToDiscard > 0)
            {
                // discard the leftmost part of the waveform that we no longer need.
                int newNumSamples = waveformRemainder.Count - samplesToDiscard;
                if (newNumSamples <= 0)
                {
                    waveformOffset += waveformRemainder.Count;
                    waveformRemainder.Clear();
                }
                else
                {
                    List<float> newRemainder = new List<float>(newNumSamples);
                    for (int i = samplesToDiscard; i < waveformRemainder.Count; i++)
                    {
                        newRemainder.Add(waveformRemainder[i]);
                    }
                    waveformOffset += samplesToDiscard;
                    waveformRemainder = newRemainder;
                }
            }
        }

        private long FirstSampleOfFrame(int numFramesNew, FrameExtractionOptions frameOpts)
        {
            return (long)(numFramesNew * frameOpts.FrameShiftMs);
        }
    }

    public interface IFeatureComputer
    {
        FrameExtractionOptions GetFrameOptions();
        int Dim();
        bool NeedRawLogEnergy();
        void Compute(float? rawLogEnergy, float vtlnWarp, List<float> signalFrame,ref float[] feature);
    }

    public class OnlineFeature : OnlineGenericBaseFeature<FbankComputer>
    {
        public OnlineFeature(FbankComputer opts) : base(opts)
        {
        }
    }
}