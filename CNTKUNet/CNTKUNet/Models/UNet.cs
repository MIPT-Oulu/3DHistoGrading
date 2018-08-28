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
        //Network functions

        //Convolution block
        private static Function ConvBlock(Variable input_var,int[] kernel_size, int in_channels, int out_channels, float[] W = null, float[] B = null, string wpath = null, string[] layer_names=null)
        {
            if(W != null & wpath != null)
            {
                throw new System.ArgumentException("1 parameter must be null!","W, wpath");
            }
            //If weight path is given, load weight from path. Otherwise use weight from float array W or, initialize new weights.
            if(wpath!=null)
            {
                W = weight_fromDisk(wpath,layer_names[0]);
                B = from_kernel(weight_fromDisk(wpath, layer_names[1]),out_channels);
            }
            //Initialize weights
            Parameter weight = make_weight(kernel_size, in_channels, out_channels, W);

            //Bias kernel size
            int[] bks = new int[kernel_size.Length];
            for (int k = 0; k<kernel_size.Length; k++)
            {
                bks[k] = 1;
            }
            //Initialize bias
            Parameter bias = make_weight(bks, out_channels, out_channels, B);

            //Generate convolution function
            Function convW = CNTKLib.Convolution(
                /*kernel and input*/ weight, input_var,
                /*strides*/ new int[] { 1, 1, in_channels},
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { true });
            //Use convolution to add bias

            Function convB = CNTKLib.Convolution(
                /*kernel and input*/ bias, convW,
                /*strides*/ new int[] { 1, 1, out_channels },
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { false });
            //ReLU
            Function relu = CNTKLib.ReLU(convB);

            return relu;
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
                /*strides*/ new int[] { 1, 1, n_channels },
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
            string[] name1 = new string[2]; string[] name2 = new string[2];
            if (wpath!=null)
            {
                name1[0] = names[0];
                name1[1] = names[1];
                name2[0] = names[2];
                name2[1] = names[3];
            }
            else
            {
                name1 = null;
                name2 = null;
            }

            //Convolution blocks
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null,null, wpath, name1);
            var block2 = ConvBlock(block1, ks, out_channels, out_channels, null,null, wpath, name2);

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
            string[] name1 = new string[2]; string[] name2 = new string[2];
            if (wpath != null)
            {
                name1[0] = names[0];
                name1[1] = names[1];
                name2[0] = names[2];
                name2[1] = names[3];
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
            var block1 = ConvBlock(catted, ks, in_channels, out_channels, null, null, wpath, name1);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, null, wpath, name2);

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
            string[] name1 = new string[2]; string[] name2 = new string[2];
            if (wpath != null)
            {
                name1[0] = names[0];
                name1[1] = names[1];
                name2[0] = names[2];
                name2[1] = names[3];
            }
            else
            {
                name1 = null;
                name2 = null;
            }

            //Convolution blocks
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, null, wpath, name1);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, null, wpath, name2);

            center = block2.Output;
        }

        //Mixer
        private static void mixer(Variable features, int[] ks, int in_channels, int out_channels,
            out Function mixer, string wpath = null, string[] names = null)
        {
            //Performs two convolutions, followed by max pooling

            //Layer names
            string[] name1 = new string[2];
            if (wpath != null)
            {
                name1[0] = names[0];
                name1[1] = names[1];
            }
            else
            {
                name1 = null;
            }

            //Convolution blocks
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, null, wpath, name1);

            mixer = block1.Output;
        }

        //Helper functions for loading the parameters

        //Generates network weights
        private static Parameter make_weight(int[] ks, int in_channels, int out_channels, float[] W = null)
        {
            //Array for dimensions
            int[] dims = new int[ks.Length + 2];
            for (int k = 0; k < ks.Length + 2; k++)
            {
                //Kernel dimensions
                if (k < ks.Length)
                {
                    dims[k] = ks[k];
                }
                //Number of feature maps
                if (k == ks.Length)
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
            if (W != null)
            {
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

        //Load weight from disk
        private static float[] weight_fromDisk(string path, string dsname)
        {
            float[] output = HDF5Loader.loadH5(path, dsname);
            return output;
        }

        //Variable declarations
        public static int BW;
        public static int[] bdims;
        public static string wpath;
        static Function model;
        static Variable feature;

        //Model initialization

        public void Initialize(int base_width, int[] batch_dims, string weight_path = null)
        {
            //Network and data parameters
            BW = base_width;
            bdims = batch_dims;
            //Path to weights
            wpath = weight_path;
            //Input feature
            feature = Variable.InputVariable(batch_dims, DataType.Float);
            //Create the model
            create_model();
        }

        //Model creation
        private static void create_model()
        {
            Console.WriteLine("Generating UNet..");
            //Parameters
            int[] ks = new int[2] { 3, 3 };

            //Encoding path

            Function down1;
            Function pooled1;
            string[] namesd1 = new string[] {"down1_0_weight", "down1_0_bias", "down1_1_weight", "down1_1_bias", };
            encoder(feature, ks, 1, BW, out down1, out pooled1, wpath, namesd1);

            Function down2;
            Function pooled2;
            string[] namesd2 = new string[] { "down2_0_weight", "down2_0_bias", "down2_1_weight", "down2_1_bias", };
            encoder(pooled1, ks, BW, 2 * BW, out down2, out pooled2, wpath, namesd2);

            Function down3;
            Function pooled3;
            string[] namesd3 = new string[] { "down3_0_weight", "down3_0_bias", "down3_1_weight", "down3_1_bias", };
            encoder(pooled2, ks, 2 * BW, 4 * BW, out down3, out pooled3, wpath, namesd3);

            Function down4;
            Function pooled4;
            string[] namesd4 = new string[] { "down4_0_weight", "down4_0_bias", "down4_1_weight", "down4_1_bias", };
            encoder(pooled3, ks, 4 * BW, 8 * BW, out down4, out pooled4, wpath, namesd4);
            
            Function down5;
            Function pooled5;
            string[] namesd5 = new string[] { "down5_0_weight", "down5_0_bias", "down5_1_weight", "down5_1_bias", };
            encoder(pooled4, ks, 8 * BW, 16 * BW, out down5, out pooled5, wpath, namesd5);

            Function down6;
            Function pooled6;
            string[] namesd6 = new string[] { "down6_0_weight", "down6_0_bias", "down6_1_weight", "down6_1_bias", };
            encoder(pooled5, ks, 16 * BW, 32 * BW, out down6, out pooled6, wpath, namesd6);

            //Center block
            Function center1;
            string[] namesc = new string[] { "center_0_weight", "center_0_bias", "center_1_weight", "center_1_bias", };
            center(pooled6, ks, 32 * BW, 32 * BW, out center1, wpath, namesc);

            //Decoding path

            Function up6;
            string[] namesu6 = new string[] { "up6_0_weight", "up6_0_bias", "up6_1_weight", "up6_1_bias", };
            decoder(center1, down6, ks, 64 * BW, 16 * BW, out up6, wpath, namesu6);

            Function up5;
            string[] namesu5 = new string[] { "up5_0_weight", "up5_0_bias", "up5_1_weight", "up5_1_bias", };
            decoder(up6, down5, ks, 32 * BW, 8 * BW, out up5, wpath, namesu5);

            Function up4;
            string[] namesu4 = new string[] { "up4_0_weight", "up4_0_bias", "up4_1_weight", "up4_1_bias", };
            decoder(up5, down4, ks, 16 * BW, 4 * BW, out up4, wpath, namesu4);

            Function up3;
            string[] namesu3 = new string[] { "up3_0_weight", "up3_0_bias", "up3_1_weight", "up3_1_bias", };
            decoder(up4, down3, ks, 8 * BW, 2 * BW, out up3, wpath, namesu3);

            Function up2;
            string[] namesu2 = new string[] { "up2_0_weight", "up2_0_bias", "up2_1_weight", "up2_1_bias", };
            decoder(up3, down2, ks, 4 * BW, 1 * BW, out up2, wpath, namesu2);
            
            Function up1;
            string[] namesu1 = new string[] { "up1_0_weight", "up1_0_bias", "up1_1_weight", "up1_1_bias", };
            decoder(up2, down1, ks, 2 * BW, 1 * BW, out up1, wpath, namesu1);

            Function unet;
            string[] namesm = new string[] { "mixer.weight", "mixer.bias" };
            mixer(up1, new int[] { 1, 1 }, BW, 1, out unet, wpath, namesm);

            model = unet;

            Console.WriteLine("Done!!");
            
        }

        //Inference
        public void Inference(float[] data)
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

    }
}
