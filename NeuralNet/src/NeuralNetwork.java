/* Alex Pendell
 * CIS 421 - Artificial Intelligence
 * Assignment 5 - ANN
 * Friday, November 22, 2019
 *
 * 
 * ASSIGNMENT 5: ARTIFICAL NEURAL NETWORK
 * This assignment we are tasked with implementing a neural network
 * that recognizes the XOR operator. That is, given two inputs the
 * network should be able to recognize (output a one (1)) if one 
 * input is a zero (0), and the other a one (1).
 * 
 *            Examples:
 * | Input 1 | Input 2 | Output |
 * |    0    |    0    |    0   |
 * |    0    |    1    |    1   |
 * |    1    |    0    |    1   |
 * |    1    |    1    |    0   |
 *
 * As illustrated, the only time the output should be one, is when
 * one (and only one) of the inputs is one, zero otherwise.
 *
 * It should be noted that For this assignment, much of my solution
 * directly inspired from the chapter eight in the textbook (both the
 * lesson, and the code example which is in C). You can find the resources
 * I used for assignment in my /resources/ directory in the root assignment
 * folder.
 */

import java.util.*;
import java.io.*;

public class NeuralNetwork {

    double RHO = 0.2;
    
    // All the nodes...
    double i0, i1, h0, h1, o0;

    // weights
    double[][] weights_ItoH = new double[2][2];
    double[] weights_HtoO = new double[2];

    // bias
    double h0_b, h1_b, o0_b;

    // final result
    int result;


    // How long do we allow it to train.
    public int epochs;
    // How often do we print the updates.
    public int update_freq;
    
    // Training data:
    int[] training_sample_one   = {0, 0, 0};
    int[] training_sample_two   = {0, 1, 1};
    int[] training_sample_three = {1, 0, 1};
    int[] training_sample_four  = {1, 1, 0};
    
    public NeuralNetwork(int time){
        
        System.out.println("Creating a new Neural Network");

        epochs = time;

        // Randomize the weights now
        randomizeWeights();
        train();

        System.out.println();
        System.out.println("FINAL STATE OF NN (POST TRAINING): ");
        printStats();
    }
    
    /* randomizeWeights()
     * This is the function that we initially call to set all of the
     * weights and biases to randomized doubles.
     * postconditions:
     *   both weight vectors will contain randomized doubles
     */
    private void randomizeWeights() {
        // Set these up so we can perform the randomization.
        // Save them here so we can change these values easily if need be
        double max = 1;
        double min = -1;

        Random r = new Random();
        
        weights_ItoH[0][0] = min + (max - min) * r.nextDouble();
        weights_ItoH[1][0] = min + (max - min) * r.nextDouble();
        weights_ItoH[0][1] = min + (max - min) * r.nextDouble();
        weights_ItoH[1][1] = min + (max - min) * r.nextDouble();

        weights_HtoO[0] = min + (max - min) * r.nextDouble();
        weights_HtoO[1] = min + (max - min) * r.nextDouble();
        
        h0_b = min + (max-min) * r.nextDouble();
        h1_b = min + (max-min) * r.nextDouble();
        o0_b = min + (max-min) * r.nextDouble();
        
    }

    /* train()
     * This is the training function for the neural network. This function
     * utilizes the training_data array to train the neural network to solve
     * xor
     * postconditions:
     *   the neural network has tweaked it's weights and biases (?) to solve
     *   xor. Nice...
     */
    private void train(){;
        for (int t = 0; t < epochs; t++){
            
            System.out.println();
            feedforward(training_sample_one);
            backpropagate(training_sample_one[2]);

            System.out.println();
            feedforward(training_sample_two);
            backpropagate(training_sample_two[2]);

            System.out.println();
            feedforward(training_sample_three);
            backpropagate(training_sample_three[2]);

            System.out.println();
            feedforward(training_sample_four);
            backpropagate(training_sample_four[2]);
            
        }
        
    }

    private void feedforward(int[] data){
        // Input layer
        i0 = data[0];
        i1 = data[1];
        
        // Hidden Layer
        h0 = sigmoid((i0 * weights_ItoH[0][0]) + (i1 * weights_ItoH[1][0]) + h0_b);
        h1 = sigmoid((i0 * weights_ItoH[0][1]) + (i1 * weights_ItoH[1][1]) + h1_b);

        // Output Layer
        o0 = sigmoid((h0 * weights_HtoO[0]) + (h1 * weights_HtoO[1]) + o0_b);
        
        if (o0 >= 0.5) result = 1;
        else result = 0;

        
        System.out.println(i0 + " XOR " + i1 + "?");
        System.out.println("o0 = " + o0 + " <- This is the unrounded answer");
        System.out.println("NN says: " + result + " =?= Expected: " + data[2]);
        if (result == data[2]) System.out.println("NN was correct!");
        else System.out.println("NN was incorrect");
        
        
        
    }

    /* backpropogate()
     * This is the function that's behind the 'magic' behind the neural
     * network. The weights are similar to sliding gauges across the neural
     * network. Some of the knobs might be far off, and some knobs might be
     * really close. The problem is that we'll need to dial each knob in by
     * a different amount. This is where the  idea of backpropogation comes
     * in.
     * precondition:
     *   The data has been fed forward in the network (i.e. feedforward() has
     *   already been executed.
     * postcondition:
     *   the neural network will have had it's weights adjusted by an amount
     *   that is proportional to how much it was wrong. 
     */ 
    private void backpropagate(int target){
        
        double out_err, h0_err, h1_err;

        // Output error
        out_err = (target - o0) * sigmoidDer(o0);

        // Hidden errors
        h0_err = (out_err * weights_HtoO[0]) * sigmoidDer(h0);
        h1_err = (out_err * weights_HtoO[0]) * sigmoidDer(h1);

        // Weight & Bias Adjustments
        weights_HtoO[0] += (RHO * out_err * h0);
        weights_HtoO[1] += (RHO * out_err * h1);
        o0_b += (RHO * out_err);

        weights_ItoH[0][0] += ((RHO * h0_err) * i0);
        weights_ItoH[1][0] += ((RHO * h0_err) * i1);
        h0_b += (RHO * h0_err);
        
        weights_ItoH[0][1] += ((RHO * h1_err) * i0);
        weights_ItoH[1][1] += ((RHO * h1_err) * i1);
        h1_b += (RHO * h1_err);
    }


    
    private double sigmoid(double value){
        return 1 / (1 + Math.exp(-value));
    }

    
    private double sigmoidDer(double value){
        return value * (1 - value);
    }
    
    private void printStats(){
        
        // Print input to hidden weights
        System.out.println("Weights (I -> H)");
        for (int i = 0; i < 2; i++){
            for (int h = 0; h < 2; h++){
                System.out.println("i" + i + "->h" + h + ": " + weights_ItoH[i][h]);
            }
        }
        
        System.out.println("Biases");
        System.out.println("h0 bias: " + h0_b);
        System.out.println("h1 bias: " + h1_b);
        System.out.println("out bias: " + o0_b);
        
        System.out.println("Weights (H -> O)");
        for (int h = 0; h < 2; h++){
                System.out.println("h" + h + "->o " + ": " + weights_HtoO[h]);
        }
        
        System.out.println("Output neuron bias");
        System.out.println("o" + o0 + " bias: " + o0_b);
    }
}
