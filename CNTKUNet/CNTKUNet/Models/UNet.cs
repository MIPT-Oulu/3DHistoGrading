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
            //Create copies of the weight array W
            NDArrayView nDArray = new NDArrayView(view, array, DeviceDescriptor.CPUDevice);
            weight.SetValue(nDArray);
            return weight;
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
                weight, input_var, new int[] { 1, 1, in_channels}));

            return conv;
        }

        //Load weight from disk *NOT TESTED*
        private static float[] weight_fromDisk(string path,string dsname)
        {
            float[] output = HDF5Loader.loadH5(path,dsname);
            return output;
        }

        //Bilinear upsampling *WEIGHTS ARE INITIALIZED INCORRECTLY, DOESN'T WORK*
        private static Function UpSampling(Variable input_var, int n_channels)
        {
            //Weight parameters for transposed convolution and smoothing
            float[] Wt = new float[2] { (float)1.0, (float)1.0 };
            int[] ksy = new int[2] { 2, 1 };
            int[] ksx = new int[2] { 1, 2 };
            float[] Ws = new float[9] { 1 / (4 * 4), 1 / (4 * 2), 1 / (4 * 4),
                                        1 / (4 * 2), 1 / (4 * 1), 1 / (4 * 2),
                                        1 / (4 * 4), 1 / (4 * 2), 1 / (4 * 4)};
            int[] kss = new int[2] { 3, 3 };
            //Initialize weights
            Parameter wtransposedy = make_weight(ksy, n_channels, n_channels, Wt);
            Parameter wtransposedx = make_weight(ksx, n_channels, n_channels, Wt);
            Parameter wsmoothing = make_weight(kss, n_channels, n_channels, Ws);
            
            //Convolutions
            Function tconvy = CNTKLib.ConvolutionTranspose(
                wtransposedy, input_var, new int[2] {2,1});

            Function tconvx = CNTKLib.ConvolutionTranspose(
                wtransposedx, tconvy, new int[2] { 1, 2 });

            Function smooth = CNTKLib.Convolution(
                wsmoothing, tconvx);

            return smooth;

        }

        //Concatenation *NOT TESTED*
        private static Function cat(Variable feature1, Variable feature2)
        {
            //Add input feature maps to variablevector
            VariableVector vec = new VariableVector();
            vec.Add(feature1);
            vec.Add(feature2);

            //Concatenate along 1st axis
            CNTK.Axis  ax = new CNTK.Axis(0);

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

        //Decoder *NOT TESTED*
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
            var upsampled = UpSampling(feature, in_channels);
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
            Console.WriteLine("Layer 1");
            //Encoding path *CONSOLE OUTPUT IS FOR DEBUGGING, RUNS WITHOUT ERRORS*
            Function down1;
            Function pooled1;
            encoder(feature, ks, 1, BW, out down1, out pooled1);
            CNTK.NDShape shape1 = pooled1.Output.Shape;
            Console.WriteLine(shape1[0]);
            Console.WriteLine(shape1[1]);
            Console.WriteLine(shape1[2]);

            Console.WriteLine("Layer 2");
            Function down2;
            Function pooled2;
            encoder(pooled1.Output, ks, BW, 2 * BW, out down2, out pooled2);
            CNTK.NDShape shape2 = pooled2.Output.Shape;
            Console.WriteLine(shape2[0]);
            Console.WriteLine(shape2[1]);
            Console.WriteLine(shape2[2]);

            Console.WriteLine("Layer 3");
            Function down3;
            Function pooled3;
            encoder(pooled2.Output, ks, 2 * BW, 4 * BW, out down3, out pooled3);
            CNTK.NDShape shape3 = pooled3.Output.Shape;
            Console.WriteLine(shape3[0]);
            Console.WriteLine(shape3[1]);
            Console.WriteLine(shape3[2]);

            Console.WriteLine("Layer 4");
            Function down4;
            Function pooled4;
            encoder(pooled3.Output, ks, 4 * BW, 8 * BW, out down4, out pooled4);
            CNTK.NDShape shape4 = pooled4.Output.Shape;
            Console.WriteLine(shape4[0]);
            Console.WriteLine(shape4[1]);
            Console.WriteLine(shape4[2]);

            Console.WriteLine("Layer 5");
            Function down5;
            Function pooled5;
            encoder(pooled4.Output, ks, 8 * BW, 16 * BW, out down5, out pooled5);
            CNTK.NDShape shape5 = pooled5.Output.Shape;
            Console.WriteLine(shape5[0]);
            Console.WriteLine(shape5[1]);
            Console.WriteLine(shape5[2]);

            
            Console.WriteLine("6");
            Function down6;
            Function pooled6;
            encoder(pooled5.Output, ks, 16 * BW, 32 * BW, out down6, out pooled6);
            
            model = pooled6;

            //Center block
            Function center1;
            center(pooled6.Output, ks, 32 * BW, 32 * BW, out center1);

            //Decoding path *UPSAMPLING FAILS*
            Function up6;
            decoder(center1, down6, ks, 64 * BW, 32 * BW, out up6);
            CNTK.NDShape shapeu6 = up6.Output.Shape;
            Console.WriteLine(shapeu6[0]);
            Console.WriteLine(shapeu6[1]);
            Console.WriteLine(shapeu6[2]);

            model = up6;
            

            Console.WriteLine("Done");

        }

        //Inference *NOT TESTED*
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
            Value featureVal;

            featureVal = Value.CreateBatch<float>(new int[] { 256, 256, 1 }, data, DeviceDescriptor.CPUDevice);

            return featureVal;
        }


    }
}
