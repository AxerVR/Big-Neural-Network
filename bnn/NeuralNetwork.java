package bnn.controller;

import helper.MyMath;
import java.util.List;

public class NeuralNetwork {

    MyMath weights_ih1, weights_h1h2, weights_h2o, bias_h1, bias_h2, bias_o;
    double l_rate = 0.01;

    public NeuralNetwork(int i, int h, int o) {
        weights_ih1 = new MyMath(h, i);
        weights_h1h2 = new MyMath(h, h);
        weights_h2o = new MyMath(o, h);

        bias_h1 = new MyMath(h,1);
        bias_h2 = new MyMath(h,1);
        bias_o = new MyMath(o,1);

    }

    public List<Double> classify(double[] A)
    {
        MyMath input = MyMath.fromArray(A);
        MyMath hidden1 = MyMath.multiply(weights_ih1, input);
        hidden1.add(bias_h1);
        hidden1.sigmoid();

        MyMath hidden2 = MyMath.multiply(weights_h1h2, hidden1);
        hidden2.add(bias_h2);
        hidden2.sigmoid();

        MyMath output = MyMath.multiply(weights_h2o, hidden2);
        output.add(bias_o);
        output.sigmoid();

        return output.toArray();
    }


    public void adjust(double[][]X, double[][]Y, int epochs)
    {
        for(int i = 0; i < epochs; i++)
        {
            int sampleN =  (int)(Math.random() * X.length);
            this.train(X[sampleN], Y[sampleN]);
        }
    }

    public void train(double [] X,double [] Y)
    {
        MyMath input = MyMath.fromArray(X);
        MyMath hidden1 = MyMath.multiply(weights_ih1, input);
        hidden1.add(bias_h1);
        hidden1.sigmoid();

        MyMath hidden2 = MyMath.multiply(weights_h1h2, hidden1);
        hidden2.add(bias_h2);
        hidden2.sigmoid();

        MyMath output = MyMath.multiply(weights_h2o,hidden2);
        output.add(bias_o);
        output.sigmoid();

        MyMath target = MyMath.fromArray(Y);

        MyMath cost = MyMath.subtract(target, output);
        MyMath gradient = output.deriveSigmoid();
        gradient.multiply(cost);
        gradient.multiply(l_rate);

        MyMath hidden2_T = MyMath.transpose(hidden2);
        MyMath wh2o_delta =  MyMath.multiply(gradient, hidden2_T);

        weights_h2o.add(wh2o_delta);
        bias_o.add(gradient);

        MyMath wh2o_T = MyMath.transpose(weights_h2o);
        MyMath hidden2_errors = MyMath.multiply(wh2o_T, cost);

        MyMath h2_gradient = hidden2.deriveSigmoid();
        h2_gradient.multiply(hidden2_errors);
        h2_gradient.multiply(l_rate);

        MyMath hidden1_T = MyMath.transpose(hidden1);
        MyMath wh1h2_delta =  MyMath.multiply(h2_gradient, hidden1_T);

        weights_h1h2.add(wh1h2_delta);
        bias_h2.add(h2_gradient);

        MyMath wh1h2_T = MyMath.transpose(weights_h1h2);
        MyMath hidden1_errors = MyMath.multiply(wh1h2_T, hidden2_errors);

        MyMath h1_gradient = hidden1.deriveSigmoid();
        h1_gradient.multiply(hidden1_errors);
        h1_gradient.multiply(l_rate);

        MyMath i_T = MyMath.transpose(input);
        MyMath wih1_delta = MyMath.multiply(h1_gradient, i_T);

        weights_ih1.add(wih1_delta);
        bias_h1.add(h1_gradient);

    }


}
