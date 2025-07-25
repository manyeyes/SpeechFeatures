// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace SpeechFeatures
{
    public class FbankComputer: IFeatureComputer
    {
        public FeatureOptions _opts { get; }
        private float _logEnergyFloor;
        private Dictionary<float, MelBanks2> _melBanks=new Dictionary<float, MelBanks2>();
        private Rfft _rfft;

        public FbankComputer(FeatureOptions opts)
        {
            _opts = opts;
            _rfft = new Rfft(opts.FrameOpts.PaddedWindowSize());

            if (opts.EnergyFloor > 0.0f)
            {
                _logEnergyFloor = (float)Math.Log(opts.EnergyFloor);
            }
            // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
            GetMelBanks(1.0f);
        }

        ~FbankComputer()
        {
            foreach (var pair in _melBanks)
            {
                _melBanks.Remove(pair.Key);
            }
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
            if (signalRawLogEnergy == null) return;
            var melBanks = GetMelBanks(vtlnWarp);
            if (signalFrame.Count != _opts.FrameOpts.PaddedWindowSize())
            {
                throw new ArgumentException("Invalid signal frame size.");
            }
            // Compute energy after window function (not the raw one).
            if (_opts.UseEnergy && !_opts.RawEnergy)
            {
                signalRawLogEnergy = (float)Math.Log(Math.Max(Utils.FeatureFunctions.InnerProduct(signalFrame.ToArray(), signalFrame.ToArray(), signalFrame.Count), float.Epsilon));
            }
            _rfft.Compute(signalFrame);  // signal_frame is modified in-place
            Utils.FeatureFunctions.ComputePowerSpectrum(signalFrame);
            // Use magnitude instead of power if requested.
            if (!_opts.UsePower)
            {
                for (int i = 0; i < signalFrame.Count / 2 + 1; i++)
                {
                    signalFrame[i] = (float)Math.Sqrt(signalFrame[i]);
                }
            }
            int melOffset = _opts.UseEnergy && !_opts.HtkCompat ? 1 : 0;
            float[] melEnergies = new float[feature.Length - melOffset];
            Array.Copy(feature, melOffset, melEnergies, 0, melEnergies.Length);
            // Sum with mel filter banks over the power spectrum
            melBanks.Compute(signalFrame.ToArray(), ref melEnergies);
            if (_opts.UseLogFbank)
            {
                for (int i = 0; i < _opts.MelOpts.numBins; i++)
                {
                    melEnergies[i] = (float)Math.Log(Math.Max(melEnergies[i], float.Epsilon));
                }
            }
            Array.Copy(melEnergies, 0, feature, melOffset, melEnergies.Length);
            // Copy energy as first value (or the last, if htk_compat == true).
            if (_opts.UseEnergy)
            {
                if (_opts.EnergyFloor > 0.0 && signalRawLogEnergy < _logEnergyFloor)
                {
                    signalRawLogEnergy = _logEnergyFloor;
                }
                int energyIndex = _opts.HtkCompat ? _opts.MelOpts.numBins : 0;
                feature[energyIndex] = (float)signalRawLogEnergy;
            }
        }

        public FrameExtractionOptions GetFrameOptions()
        {
            return _opts.FrameOpts;
        }
    }
}