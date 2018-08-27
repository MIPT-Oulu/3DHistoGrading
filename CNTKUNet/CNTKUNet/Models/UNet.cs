using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using CNTKUNet.Components;

namespace CNTKUNet.Models
{
    class UNet
    {
        //Function Declarations

        //Generates network weights
        private static Parameter make_weight(int[] ks, int in_channels, int out_channels, float[] W = null)
        {
            //Array for dimensions
            int[] dims = new int[ks.Length + 2];
            for(int k=0;k<ks.Length+2;k++)
            {
                //Kernel dimensions
                if (k < ks.Length)
                {
                    dims[k] = ks[k];
                }
                //Number of feature maps
                if(k == ks.Length)
                {
                    dims[k] = in_channels;
                }
                if (k == ks.Length + 1)
                {
                    dims[k] = out_channels;
                }
            }            

            //Initialize new weight
            Parameter weight = new Parameter(
                dims, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.CPUDevice);

            //Set weight values from float array if given
            if(W != null)
            {
                Console.WriteLine("{0},{1},{2},{3}", dims[0], dims[1], dims[2], dims[3]);
                weight = weight_fromFloat(weight, W, dims);
            }            

            return weight;
        }

        //Load weights from array
        private static Parameter weight_fromFloat(Parameter weight, float[] array, int[] view)
        {
            //Generate weight array with correct dimensions
            NDArrayView nDArray = new NDArrayView(view, array, DeviceDescriptor.CPUDevice);
            weight.SetValue(nDArray);
            return weight;
        }

        //Generate weight array with correct dimensions from given input kernel
        private static float[] from_kernel(float[] kernel, int dim)
        {
            //Total length of new weight array
            int K = dim * dim * kernel.Length;
            //Empty array for output
            float[] outarray = new float[K];
            //Loop over output array
            for (int k = 0; k < K; k += kernel.Length * dim + kernel.Length)
            {
                //Loop over input array
                for (int kk = 0; kk < kernel.Length; kk++)
                {
                    outarray[k + kk] = kernel[kk];
                }
            }
            //Return the array
            return outarray;
        }
            //Convolution block
        private static Function ConvBlock(Variable input_var,int[] kernel_size, int in_channels, int out_channels, float[] W = null, string wpath = null, string layer_name=null)
        {
            if(W != null & wpath != null)
            {
                throw new System.ArgumentException("1 parameter must be null!","W, wpath");
            }
            //If weight path is given, load weight from path. Otherwise use weight from float array W or, initialize new weights.
            if(wpath!=null)
            {
                W = weight_fromDisk(wpath,layer_name);
            }
            //Initialize weights
            Parameter weight = make_weight(kernel_size, in_channels, out_channels, W);

            //Generate convolution function
            Function conv = CNTKLib.ReLU(
                CNTKLib.Convolution(
                /*kernel and input*/ weight, input_var,
                /*strides*/ new int[] { 1, 1, in_channels},
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { true }));

            return conv;
        }

        //Load weight from disk *NOT TESTED*
        private static float[] weight_fromDisk(string path,string dsname)
        {
            float[] output = HDF5Loader.loadH5(path,dsname);
            return output;
        }

        //Bilinear upsampling
        private static Function UpSampling(Variable input_var, int n_channels)
        {
            //Weight parameters for transposed convolution and smoothing
            float[] Wt = new float[2] { (float)1.0, (float)1.0 };
            int[] ksy = new int[2] { 2, 1 };
            int[] ksx = new int[2] { 1, 2 };
            float[] Ws = new float[9] { (float)1 / (4 * 4), (float)1 / (4 * 2), (float)1 / (4 * 4),
                                        (float)1 / (4 * 2), (float)1 / (4 * 1), (float)1 / (4 * 2),
                                        (float)1 / (4 * 4), (float)1 / (4 * 2), (float)1 / (4 * 4)};
            int[] kss = new int[2] { 3, 3 };
            //Initialize weights
            Parameter wtransposedy = make_weight(ksy, n_channels, n_channels, from_kernel(Wt,n_channels));
            Parameter wtransposedx = make_weight(ksx, n_channels, n_channels, from_kernel(Wt,n_channels));
            Parameter wsmoothing = make_weight(kss, n_channels, n_channels, from_kernel(Ws,n_channels));

            //Transposed convolutions
            Function tconvy = CNTKLib.ConvolutionTranspose(
                /*kernel and input*/ wtransposedy, input_var,
                /*strides*/ new int[] { 2, 1, n_channels },
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { false });
            Function tconvx = CNTKLib.ConvolutionTranspose(
                /*kernel and input*/ wtransposedx, tconvy,
                /*strides*/ new int[] { 1, 2, n_channels },
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { false });
            //Smoothing
            Function smooth = CNTKLib.Convolution(
                /*kernel and input*/ wsmoothing, tconvx,
                /*strides*/ new int[] { 1, 1 },
                /*sharing*/new bool[] { true },
                /*padding*/ new bool[] { true });
            
            return smooth;

        }

        //Concatenation
        private static Function cat(Variable feature1, Variable feature2)
        {
            //Add input feature maps to variablevector
            VariableVector vec = new VariableVector();
            vec.Add(feature1);
            vec.Add(feature2);

            //Concatenate along last axis
            Axis ax = new Axis(2);

            return CNTKLib.Splice(vec, ax);
        }

        
        //Encoder
        private static void encoder(Variable features, int[] ks, int in_channels, int out_channels,
            out Function down, out Function pooled, string wpath = null, string[] names = null)
        {
            //Performs two convolutions, followed by max pooling

            //Layer names
            string name1; string name2;
            if(wpath!=null)
            {
                name1 = names[0];
                name2 = names[1];
            }
            else
            {
                name1 = null;
                name2 = null;
            }

            //Convolution blocks
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, wpath, name1);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, wpath, name2);

            //Max pooling
            var pooling = CNTKLib.Pooling(
                block2.Output, PoolingType.Max,
                new int[] { 2, 2 }, new int[] { 2, 2 },
                new bool[] { true });

            down = block2.Output;
            pooled = pooling.Output;
        }

        //Decoder
        private static void decoder(Variable feature, Variable feature_big, int[] ks, int in_channels, int out_channels,
            out Function up, string wpath = null, string[] names = null)
        {
            //Layer names
            string name1; string name2;
            if (wpath != null)
            {
                name1 = names[0];
                name2 = names[1];
            }
            else
            {
                name1 = null;
                name2 = null;
            }
            //Concatenation and upsampling
            var upsampled = UpSampling(feature, in_channels/2);
            var catted = cat(upsampled.Output, feature_big);
            //Convolutions
            var block1 = ConvBlock(catted, ks, in_channels, out_channels, null, wpath, name1);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, wpath, name2);

            //Max pooling
            var pooling = CNTKLib.Pooling(
                block2.Output, PoolingType.Max,
                new int[] { 2, 2 }, new int[] { 2, 2 },
                new bool[] { true });

            up = block2.Output;
        }

        //Center block
        private static void center(Variable features, int[] ks, int in_channels, int out_channels,
            out Function center, string wpath = null, string[] names = null)
        {
            //Performs two convolutions, followed by max pooling

            //Layer names
            string name1; string name2;
            if (wpath != null)
            {
                name1 = names[0];
                name2 = names[1];
            }
            else
            {
                name1 = null;
                name2 = null;
            }

            //Convolution blocks
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, wpath, name1);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, wpath, name2);

            center = block2.Output;
        }

        //Variable declarations
        public static int BW;
        public static int[] bdims;
        public static string wpath;
        static Function model;
        static Variable feature;

        //Initialization

        public void Initialize(int base_width, int[] batch_dims, string weight_path = null)
        {
            BW = base_width;
            bdims = batch_dims;
            wpath = weight_path;

            feature = Variable.InputVariable(batch_dims, DataType.Float);

            //Create the model
            create_model();
        }


        private static void create_model()
        {
            //Parameters
            int[] ks = new int[2] { 3, 3 };

            //Encoding path

            Function down1;
            Function pooled1;
            encoder(feature, ks, 1, BW, out down1, out pooled1);
            CNTK.NDShape shape1 = pooled1.Output.Shape;

            Function down2;
            Function pooled2;
            encoder(pooled1.Output, ks, BW, 2 * BW, out down2, out pooled2);

            Function down3;
            Function pooled3;
            encoder(pooled2.Output, ks, 2 * BW, 4 * BW, out down3, out pooled3);

            Function down4;
            Function pooled4;
            encoder(pooled3.Output, ks, 4 * BW, 8 * BW, out down4, out pooled4);

            Function down5;
            Function pooled5;
            encoder(pooled4.Output, ks, 8 * BW, 16 * BW, out down5, out pooled5);
            
            Function down6;
            Function pooled6;
            encoder(pooled5.Output, ks, 16 * BW, 32 * BW, out down6, out pooled6);

            //Center block
            Function center1;
            center(pooled6.Output, ks, 32 * BW, 32 * BW, out center1);

            //Decoding path
            Function up6;
            decoder(center1, down6, ks, 64 * BW, 16 * BW, out up6);

            Function up5;
            decoder(up6, down5, ks, 32 * BW, 8 * BW, out up5);

            Function up4;
            decoder(up5, down4, ks, 16 * BW, 4 * BW, out up4);

            Function up3;
            decoder(up4, down3, ks, 8 * BW, 2 * BW, out up3);

            Function up2;
            decoder(up3, down2, ks, 4 * BW, 1 * BW, out up2);

            Function up1;
            decoder(up2, down1, ks, 2 * BW, 1 * BW, out up1);

            model = up1;
            

            Console.WriteLine("Done");

        }

        //Inference *NOT TESTED*
        public void Inference(float[] data, int[] dims)
        {
            //Generate batch from input data
            Value inputdata = mapBatch(data);
            //Map input array to feature
            var inputDataMap = new Dictionary<Variable, Value>() { { feature, inputdata } };
            //Create output featuremap
            var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };
            //Forward pass
            model.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
        }

        private static Value mapBatch(float[] data)
        {
            //Map input data to value
            Value featureVal = Value.CreateBatch<float>(bdims, data, DeviceDescriptor.CPUDevice);

            return featureVal;
        }

        //FOR DEBUGGING UPSAMPLING
        public float[] bilinear_upsampling(float[] data, int[] dims)
        {
            //Input feature
            Variable feature = Variable.InputVariable(dims, DataType.Float);
            //Upsampling method
            Function sampler = UpSampling(feature, 1);

            //Feature maps for input and output
            Value inputValue = Value.CreateBatch<float>(dims, data, DeviceDescriptor.CPUDevice);

            Dictionary<Variable,Value> inputmap = new Dictionary<Variable, Value> { { feature, inputValue } };
            Dictionary<Variable, Value> outputmap = new Dictionary<Variable, Value> { { sampler.Output, null } };

            //Evaluate upsampling
            sampler.Evaluate(inputmap,outputmap, DeviceDescriptor.CPUDevice);

            //Get output to list
            Value outval = outputmap[sampler.Output];

            NDShape outshape = outval.Shape;

            IList<IList<float>> outdata = outval.GetDenseData<float>(sampler.Output);

            IList<float> outlist = outdata.First();

            //Collect output to float array
            float[] outarray = new float[outshape[0]*outshape[1]*outshape[2]];

            int k = 0;
            foreach (float item in outlist)
            {
                outarray[k] = item;
                k++;
            }

            return outarray;
        }


    }
}
