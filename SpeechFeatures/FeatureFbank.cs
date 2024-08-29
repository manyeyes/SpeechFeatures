// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace SpeechFeatures
{
    public struct FbankOptions
    {
        public FrameExtractionOptions FrameOpts  = new FrameExtractionOptions();
        // append an extra dimension with energy to the filter banks
        public MelBanksOptions MelOpts  = new MelBanksOptions();
        public bool UseEnergy   = false;
        public float EnergyFloor   = 0.0f;  // active iff use_energy==true
        // If true, compute log_energy before preemphasis and windowing
        // If false, compute log_energy after preemphasis ans windowing
        public bool RawEnergy   = true;  // active iff use_energy==true
        // If true, put energy last (if using energy)
        // If false, put energy first
        public bool HtkCompat   = false;  // active iff use_energy==true

        // if true (default), produce log-filterbank, else linear
        public bool UseLogFbank   = true;
        // if true (default), use power in filterbank
        // analysis, else magnitude.
        public bool UsePower   = true;
        public FbankOptions()
        {
            MelOpts.numBins = 23;
        }
    }

    public class FbankComputer: IFeatureComputer
    {
        public FbankOptions Opts { get; }
        private float logEnergyFloor;
        private Dictionary<float, MelBanks> melBanks=new Dictionary<float, MelBanks>();
        private Rfft rfft;

        public FbankComputer(FbankOptions opts)
        {
            Opts = opts;
            rfft = new Rfft(opts.FrameOpts.PaddedWindowSize());
            if (opts.EnergyFloor > 0.0f)
            {
                logEnergyFloor = (float)Math.Log(opts.EnergyFloor);
            }
            // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
            GetMelBanks(1.0f);
        }

        ~FbankComputer()
        {
            foreach (var pair in melBanks)
            {
                melBanks.Remove(pair.Key);
            }
        }

        private MelBanks GetMelBanks(float vtlnWarp)
        {
            MelBanks? thisMelBanks = null;
            if (!melBanks.TryGetValue(vtlnWarp, out thisMelBanks))
            {
                thisMelBanks = new MelBanks(Opts.MelOpts, Opts.FrameOpts, vtlnWarp);
                melBanks[vtlnWarp] = thisMelBanks;
            }
            return thisMelBanks;
        }

        public int Dim()
        {
            return Opts.MelOpts.numBins + (Opts.UseEnergy ? 1 : 0);
        }

        public bool NeedRawLogEnergy()
        {
            return Opts.UseEnergy && Opts.RawEnergy;
        }
        public void Compute(float? signalRawLogEnergy, float vtlnWarp, List<float> signalFrame,ref float[] feature)
        {
            if (signalRawLogEnergy == null) return;
            var melBanks = GetMelBanks(vtlnWarp);
            if (signalFrame.Count != Opts.FrameOpts.PaddedWindowSize())
            {
                throw new ArgumentException("Invalid signal frame size.");
            }
            // Compute energy after window function (not the raw one).
            if (Opts.UseEnergy && !Opts.RawEnergy)
            {
                signalRawLogEnergy = (float)Math.Log(Math.Max(Utils.FeatureFunctions.InnerProduct(signalFrame.ToArray(), signalFrame.ToArray(), signalFrame.Count), float.Epsilon));
            }
            rfft.Compute(signalFrame);  // signal_frame is modified in-place
            Utils.FeatureFunctions.ComputePowerSpectrum(signalFrame);
            // Use magnitude instead of power if requested.
            if (!Opts.UsePower)
            {
                for (int i = 0; i < signalFrame.Count / 2 + 1; i++)
                {
                    signalFrame[i] = (float)Math.Sqrt(signalFrame[i]);
                }
            }
            int melOffset = Opts.UseEnergy && !Opts.HtkCompat ? 1 : 0;
            float[] melEnergies = new float[feature.Length - melOffset];
            Array.Copy(feature, melOffset, melEnergies, 0, melEnergies.Length);
            // Sum with mel filter banks over the power spectrum
            melBanks.Compute(signalFrame.ToArray(), ref melEnergies);
            if (Opts.UseLogFbank)
            {
                for (int i = 0; i < Opts.MelOpts.numBins; i++)
                {
                    melEnergies[i] = (float)Math.Log(Math.Max(melEnergies[i], float.Epsilon));
                }
            }            
            Array.Copy(melEnergies, 0, feature, melOffset, melEnergies.Length);
            // Copy energy as first value (or the last, if htk_compat == true).
            if (Opts.UseEnergy)
            {
                if (Opts.EnergyFloor > 0.0 && signalRawLogEnergy < logEnergyFloor)
                {
                    signalRawLogEnergy = logEnergyFloor;
                }
                int energyIndex = Opts.HtkCompat ? Opts.MelOpts.numBins : 0;
                feature[energyIndex] = (float)signalRawLogEnergy;
            }
        }

        public FrameExtractionOptions GetFrameOptions()
        {
            return Opts.FrameOpts;
        }
    }
}