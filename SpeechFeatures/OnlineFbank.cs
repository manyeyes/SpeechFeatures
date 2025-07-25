// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace SpeechFeatures
{
    public class OnlineFbank : IDisposable
    {
        internal bool _disposed = false;
        internal FeatureOptions _opts = new FeatureOptions();
        internal IFeature _feature;
        private float _sample_rate = 16000.0F;
        private int _num_bins = 80;
        private int _last_frame_index = 0;

        /// <summary>
        /// OnlineFbank
        /// </summary>
        /// <param name="dither"></param>
        /// <param name="snip_edges"></param>
        /// <param name="sample_rate"></param>
        /// <param name="num_bins"></param>
        /// <param name="frame_shift"></param>
        /// <param name="frame_length"></param>
        /// <param name="energy_floor"></param>
        /// <param name="debug_mel"></param>
        /// <param name="window_type">window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')</param>
        /// <param name="feature_type">feature_type (string): Type of feature ('whisper'|'fbank')</param>
        public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming", string feature_type = "fbank")
        {
            _sample_rate = sample_rate;
            _num_bins = num_bins;
            this._opts = FeatureShip.GetFeatureOptions(
                 dither: dither,
                 snip_edges: snip_edges,
                 sample_rate: sample_rate,
                 num_bins: num_bins,
                 frame_shift: frame_shift,
                 frame_length: frame_length,
                 energy_floor: energy_floor,
                 debug_mel: debug_mel,
                 window_type: window_type
                 );
            this._feature = FeatureShip.GetOnlineFeature(this._opts, featureType: feature_type);
        }

        /// <summary>
        /// Get one frame at a time
        /// </summary>
        /// <param name="samples"></param>
        /// <returns></returns>
        public float[] GetFbank(float[] samples)
        {
            FeatureShip.AcceptWaveform(_feature, _sample_rate, samples, samples.Length);
            int framesNum = FeatureShip.GetNumFramesReady(_feature);
            int n = framesNum - _last_frame_index;
            float[] fbanks = new float[n * _num_bins];
            for (int i = _last_frame_index; i < framesNum; i++)
            {
                FeatureData featureData = new FeatureData();
                FeatureShip.GetFeature(_feature, i, ref featureData);
                float[] _featureData = new float[featureData.data_length];
                _featureData = featureData.data;
                Array.Copy(_featureData, 0, fbanks, (i - _last_frame_index) * _num_bins, _featureData.Length);
                featureData.data = null;
                _featureData = null;
            }
            _last_frame_index += n;
            samples = null;
            return fbanks;
        }

        /// <summary>
        /// Get all current frames each time
        /// Faster than GetFbank
        /// </summary>
        /// <param name="samples"></param>
        /// <returns></returns>
        public float[] GetFbankIndoor(float[] samples)
        {
            FeatureShip.AcceptWaveform(_feature, _sample_rate, samples, samples.Length);
            int framesNum = FeatureShip.GetNumFramesReady(_feature);
            FeatureDatas featureDatas = new FeatureDatas();
            FeatureShip.GetFeatures(_feature, _last_frame_index, ref featureDatas);
            float[] _fbankDatas = new float[featureDatas.data_length];
            _fbankDatas = featureDatas.data;
            //Marshal.Copy(fbankDatas.data, _fbankDatas, 0, fbankDatas.data_length);
            _last_frame_index = framesNum - 1;
            samples = null;
            return _fbankDatas;
        }
        public void InputFinished()
        {
            FeatureShip.InputFinished(_feature);
        }


        protected virtual void Dispose(bool disposing)
        {
            if (!disposing)
            {
                if (_feature != null)
                {
                    _feature = null;
                }
                this._disposed = true;
            }
        }
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        ~OnlineFbank()
        {
            Dispose(this._disposed);
        }
    }
}