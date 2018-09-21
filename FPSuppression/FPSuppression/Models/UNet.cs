using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using HistoGrading.Components;

namespace HistoGrading.Models
{
    class UNet
    {
        //Network functions

        //Convolution block
        private static Function ConvBlock(Variable input_var, int[] kernel_size, int in_channels, int out_channels, float[] W = null, float[] B = null,
            string wpath = null, string[] layer_names = null, bool padding = true, bool use_relu = true, bool use_bn = true)
        {
            Function output_function;

            if (W != null & wpath != null)
            {
                throw new System.ArgumentException("1 parameter must be null!", "W, wpath");
            }
            //If weight path is given, load weight from path. Otherwise use weight from float array W or, initialize new weights.
            if (wpath != null)
            {
                W = weight_fromDisk(wpath, layer_names[0]);
            }
            //Initialize weights
            Parameter weight = make_weight(kernel_size, in_channels, out_channels, W, layer_names[0]);

            //Generate convolution function
            Function convW = CNTKLib.Convolution(
                /*kernel and input*/ weight, input_var,
                /*strides*/ new int[] { 1, 1, in_channels },
                /*sharing*/ new CNTK.BoolVector { true },
                /*padding*/ new CNTK.BoolVector { padding });

            //Initialize bias
            if (wpath != null)
            {
                B = make_copies(weight_fromDisk(wpath, layer_names[1]), convW.Output.Shape);
            }
            Parameter bias = make_bias(convW.Output.Shape, B, layer_names[1]);

            //Add bias
            Function add = CNTKLib.Plus(convW, bias);

            //Sigmoid
            Function sig = CNTKLib.Sigmoid(add);

            if (use_bn == true)
            {
                //Initialize batch normalization
                int[] bns = new int[] { 1, 1 };
                Parameter scale;
                Parameter bnbias;
                Parameter rm;
                Parameter rv;
                var n = Constant.Scalar(0.0f, DeviceDescriptor.GPUDevice(0));

                make_bn_pars(out_channels, add.Output.Shape, out scale, out bnbias, out rm, out rv, wpath, layer_names[0]);

                //Batch normalization
                Function bn = CNTKLib.BatchNormalization(add, scale, bnbias, rm, rv, n, true);

                //ReLU
                Function relu = CNTKLib.ReLU(bn);
                output_function = relu;
            }
            else
            {
                if (use_relu == true)
                {
                    //ReLU
                    Function relu = CNTKLib.ReLU(add);
                    output_function = relu;
                }
                else
                {
                    output_function = sig;
                }
            }

            return output_function;
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
            Parameter wtransposedy = make_weight(ksy, n_channels, n_channels, from_kernel(Wt, n_channels), "kernel_y");
            Parameter wtransposedx = make_weight(ksx, n_channels, n_channels, from_kernel(Wt, n_channels), "kernel_x");
            Parameter wsmoothing = make_weight(kss, n_channels, n_channels, from_kernel(Ws, n_channels), "kernel_smoothing");

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
            out Function down, out Function pooled, string wpath = null, string[] names = null, bool bn = true)
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
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, null, wpath, name1, true, true, bn);
            var block2 = ConvBlock(block1, ks, out_channels, out_channels, null, null, wpath, name2, true, true, bn);

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
            out Function up, string wpath = null, string[] names = null, bool bn = true)
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
            var upsampled = UpSampling(feature, in_channels / 2);
            var catted = cat(upsampled.Output, feature_big);
            //Convolutions
            var block1 = ConvBlock(catted, ks, in_channels, out_channels, null, null, wpath, name1, true, true, bn);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, null, wpath, name2, true, true, bn);

            //Max pooling
            var pooling = CNTKLib.Pooling(
                block2.Output, PoolingType.Max,
                new int[] { 2, 2 }, new int[] { 2, 2 },
                new bool[] { true });

            up = block2.Output;
        }

        //Center block
        private static void center(Variable features, int[] ks, int in_channels, int out_channels,
            out Function center, string wpath = null, string[] names = null, bool bn = false)
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
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, null, wpath, name1, true, true, bn);
            var block2 = ConvBlock(block1.Output, ks, out_channels, out_channels, null, null, wpath, name2, true, true, bn);

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
            var block1 = ConvBlock(features, ks, in_channels, out_channels, null, null, wpath, name1, true, false, false);

            mixer = block1.Output;
        }

        //Helper functions for loading the parameters

        //Generates network weights
        private static Parameter make_weight(int[] ks, int in_channels, int out_channels, float[] W = null, string name = null)
        {
            Parameter weight;
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

            //Set weight values from float array if given
            if (W != null)
            {
                //Initialize new weight
                weight = new Parameter(dims, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0), name);
                weight = weight_fromFloat(weight, W, dims);
            }
            else
            {
                //Initialize new weight
                weight = new Parameter(dims, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
            }

            return weight;
        }

        //Generates network bias
        private static Parameter make_bias(NDShape dims, float[] W = null, string name = null)
        {
            Parameter bias;
            //Make int array of dims
            int[] bks = new int[dims.Rank];
            for (int k = 0; k < bks.Length; k++)
            {
                bks[k] = dims[k];
            }
            //Set weight values from float array if given
            if (W != null)
            {
                //Initialize new weight
                bias = new Parameter(dims, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0), name);
                bias = weight_fromFloat(bias, W, bks);
            }
            else
            {
                bias = new Parameter(dims, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
            }

            return bias;
        }

        //Generates batch normalization parameters
        private static void make_bn_pars(int out_channels, NDShape shape, out Parameter w, out Parameter b, out Parameter m, out Parameter v, string wpath = null, string layer_name = null)
        {
            //Set weight values from float array if given
            if (wpath != null)
            {
                //Generate parameter names
                string[] S = layer_name.Split('_');
                string wn = "bn" + S[0] + "_" + S[1] + "_" + "weight";
                string bn = "bn" + S[0] + "_" + S[1] + "_" + "bias";
                string mn = "bn" + S[0] + "_" + S[1] + "_" + "running_mean";
                string vn = "bn" + S[0] + "_" + S[1] + "_" + "running_var";
                //Initialize new weight
                w = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                float[] W = make_copies(weight_fromDisk(wpath, wn), shape, true);
                w = weight_fromFloat(w, W, new int[] { out_channels });
                //Initialize new bias
                b = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                W = make_copies(weight_fromDisk(wpath, bn), shape, true);
                b = weight_fromFloat(b, W, new int[] { out_channels });
                //Initialize new running mean
                m = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                W = make_copies(weight_fromDisk(wpath, mn), shape, true);
                m = weight_fromFloat(m, W, new int[] { out_channels });
                //Initialize new variance
                v = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                W = make_copies(weight_fromDisk(wpath, vn), shape, true);
                v = weight_fromFloat(v, W, new int[] { out_channels });
            }
            else
            {
                //Initialize new weight
                w = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                //Initialize new bias
                b = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                //Initialize new running mean
                m = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
                //Initialize new variance
                v = new Parameter(new int[] { out_channels }, DataType.Float, CNTKLib.GlorotUniformInitializer(), DeviceDescriptor.GPUDevice(0));
            }
        }

        //Load weights from array
        private static Parameter weight_fromFloat(Parameter weight, float[] array, int[] view)
        {
            //Generate weight array with correct dimensions
            NDArrayView nDArray = new NDArrayView(view, array, DeviceDescriptor.GPUDevice(0));
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

        //Copy bias to match the dimensions of the input
        private static float[] make_copies(float[] kernel, NDShape dims, bool swap_axes = false)
        {
            int K = dims[0] * dims[1];
            float[] outarray = new float[dims[0] * dims[1] * dims[2]];
            if (swap_axes == false)
            {
                //Loop over number of maps
                for (int k = 0; k < dims[2]; k++)
                {
                    //Loop over spatial dimensions
                    for (int kk = 0; kk < K; kk++)
                    {
                        outarray[k * K + kk] = kernel[k];
                    }
                }
            }
            else
            {
                //Loop over number of maps
                for (int k = 0; k < K; k++)
                {
                    //Loop over spatial dimensions
                    for (int kk = 0; kk < dims[2]; kk++)
                    {
                        outarray[k * dims[2] + kk] = kernel[kk];
                    }
                }
            }
            return outarray;
        }

        //Load weight from disk
        private static float[] weight_fromDisk(string path, string dsname)
        {
            float[] output = HDF5Loader.loadH5(path, dsname);
            return output;
        }

        //Print parameter
        private static void print_parameter(Function input)
        {
            IList<Parameter> pars = input.Parameters();
            foreach (Parameter par in pars)
            {
                string name = par.Name;
                Console.WriteLine(name);
                NDArrayView view = par.Value();
                Value wv = new Value(view);
                IList<IList<float>> wd = wv.GetDenseData<float>(par);
                foreach (IList<float> L in wd)
                {
                    foreach (float v in L)
                    {
                        Console.WriteLine(v);
                    }
                }
                Console.ReadKey();
            }
        }

        //Variable declarations
        public static int BW;
        public static int[] input_size;
        public static int n_samples;
        public static string wpath;
        static Function model;
        static Variable feature;

        //Model initialization

        public void Initialize(int base_width, int[] sample_size, string weight_path = null, bool use_bn = true)
        {
            //Network and data parameters
            BW = base_width;
            input_size = sample_size;
            //Path to weights
            wpath = weight_path;
            //Input feature
            feature = Variable.InputVariable(input_size, DataType.Float);
            //Create the model
            create_model(use_bn);
        }

        //Model creation
        private static void create_model(bool use_bn = false)
        {
            //Parameters
            int[] ks = new int[2] { 3, 3 };

            //Encoding path

            Function down1;
            Function pooled1;
            string[] namesd1 = new string[] { "down1_0_weight", "down1_0_bias", "down1_1_weight", "down1_1_bias", };
            encoder(feature, ks, 1, BW, out down1, out pooled1, wpath, namesd1, use_bn);


            Function down2;
            Function pooled2;
            string[] namesd2 = new string[] { "down2_0_weight", "down2_0_bias", "down2_1_weight", "down2_1_bias", };
            encoder(pooled1, ks, BW, 2 * BW, out down2, out pooled2, wpath, namesd2, use_bn);

            Function down3;
            Function pooled3;
            string[] namesd3 = new string[] { "down3_0_weight", "down3_0_bias", "down3_1_weight", "down3_1_bias", };
            encoder(pooled2, ks, 2 * BW, 4 * BW, out down3, out pooled3, wpath, namesd3, use_bn);

            Function down4;
            Function pooled4;
            string[] namesd4 = new string[] { "down4_0_weight", "down4_0_bias", "down4_1_weight", "down4_1_bias", };
            encoder(pooled3, ks, 4 * BW, 8 * BW, out down4, out pooled4, wpath, namesd4, use_bn);

            Function down5;
            Function pooled5;
            string[] namesd5 = new string[] { "down5_0_weight", "down5_0_bias", "down5_1_weight", "down5_1_bias", };
            encoder(pooled4, ks, 8 * BW, 16 * BW, out down5, out pooled5, wpath, namesd5, use_bn);

            Function down6;
            Function pooled6;
            string[] namesd6 = new string[] { "down6_0_weight", "down6_0_bias", "down6_1_weight", "down6_1_bias", };
            encoder(pooled5, ks, 16 * BW, 32 * BW, out down6, out pooled6, wpath, namesd6, use_bn);

            //Center block
            Function center1;
            string[] namesc = new string[] { "center_0_weight", "center_0_bias", "center_1_weight", "center_1_bias", };
            center(pooled6, ks, 32 * BW, 32 * BW, out center1, wpath, namesc, use_bn);

            //Decoding path

            Function up6;
            string[] namesu6 = new string[] { "up6_0_weight", "up6_0_bias", "up6_1_weight", "up6_1_bias", };
            decoder(center1, down6, ks, 64 * BW, 16 * BW, out up6, wpath, namesu6, use_bn);

            Function up5;
            string[] namesu5 = new string[] { "up5_0_weight", "up5_0_bias", "up5_1_weight", "up5_1_bias", };
            decoder(up6, down5, ks, 32 * BW, 8 * BW, out up5, wpath, namesu5, use_bn);

            Function up4;
            string[] namesu4 = new string[] { "up4_0_weight", "up4_0_bias", "up4_1_weight", "up4_1_bias", };
            decoder(up5, down4, ks, 16 * BW, 4 * BW, out up4, wpath, namesu4, use_bn);

            Function up3;
            string[] namesu3 = new string[] { "up3_0_weight", "up3_0_bias", "up3_1_weight", "up3_1_bias", };
            decoder(up4, down3, ks, 8 * BW, 2 * BW, out up3, wpath, namesu3, use_bn);

            Function up2;
            string[] namesu2 = new string[] { "up2_0_weight", "up2_0_bias", "up2_1_weight", "up2_1_bias", };
            decoder(up3, down2, ks, 4 * BW, 1 * BW, out up2, wpath, namesu2, use_bn);

            Function up1;
            string[] namesu1 = new string[] { "up1_0_weight", "up1_0_bias", "up1_1_weight", "up1_1_bias", };
            decoder(up2, down1, ks, 2 * BW, 1 * BW, out up1, wpath, namesu1, use_bn);

            //Output layer
            Function unet;
            string[] namesm = new string[] { "mixer.weight", "mixer.bias" };
            mixer(up1, new int[] { 1, 1 }, BW, 1, out unet, wpath, namesm);

            model = unet;
        }

        //Inference
        public IList<IList<float>> Inference(float[] data)
        {
            //Get number of samples
            n_samples = data.Length / (input_size[0] * input_size[1]);
            //Generate batch from input data
            Value inputdata = mapBatch(data, n_samples);
            //Map input array to feature
            var inputDataMap = new Dictionary<Variable, Value>() { { feature, inputdata } };
            //Create output featuremap
            var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };
            //Forward pass
            model.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.GPUDevice(0));
            //Get output
            IList<IList<float>> output = get_output(outputDataMap, input_size, n_samples);

            return output;
        }

        //Method for mapping float array to minibatch
        private static Value mapBatch(float[] data, int n_inputs)
        {
            //Map input data to value
            Value featureVal = Value.CreateBatch<float>(input_size, data, 0, data.Length, DeviceDescriptor.GPUDevice(0));

            return featureVal;
        }

        //Method for extracting inference output
        private static IList<IList<float>> get_output(Dictionary<Variable, Value> result, int[] dims, int samples)
        {
            //Get dictionary keys
            Dictionary<Variable, Value>.KeyCollection keys = result.Keys;
            //New nested list for output
            IList<IList<float>> outlist;
            //Get first key
            Variable key = keys.First();
            //Data to list
            var D = result[key];
            outlist = D.GetDenseData<float>(key);

            /*
            //Output array
            float[] outarray = new float[dims[0] * dims[1] * samples];
            //Iterator
            int c = 0;
            //Iterate over list elements and collect to array
            foreach (IList<float> item in outlist)
            {
                foreach (float k in item)
                {
                    outarray[c] = k;
                    //Increase iterator
                    c += 1;
                }
            }
            */
            return outlist;
        }

    }
}