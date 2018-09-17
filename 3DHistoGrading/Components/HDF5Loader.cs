using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using HDF5DotNet;

namespace HistoGrading.Components
{
    class HDF5Loader
    {
        //Load weights from hdf5 file. Weights must be saved as a vector per layer
        public static float[] loadH5(string path, string dsname)
        {
            //Get file id
            var h5fid = H5F.open(path, H5F.OpenMode.ACC_RDONLY);
            //Get dataset id
            var h5did = H5D.open(h5fid, dsname);
            //Dataset size
            var h5space = H5D.getSpace(h5did);
            var h5size = H5S.getSimpleExtentDims(h5space);

            //Dataset size to array
            var S = h5size.ToArray();

            //Empty double array for the data
            double[] data = new double[S[0]];

            //Read the dataset

            var h5array = new H5Array<double>(data);
            var h5dtype = H5D.getType(h5did);

            H5D.read(h5did, h5dtype, h5array);

            //Convert to float
            float[] newarray = new float[data.Length];

            Parallel.For(0, data.Length, (k) =>
            {
                newarray[k] = (float)data[k];
            });

            return newarray;
        }
    }
}