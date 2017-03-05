package multilayer.perceptron.weka;

import java.io.FileReader;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author supun perera
 */
public class MultilayerPerceptronWeka {

    public static void main(String[] args) {
        
        MultilayerPerceptronWeka mlp = new MultilayerPerceptronWeka();
                
        String path = "F:\\My_work_place\\NetBeansProjects\\Multilayer Perceptron Weka\\data\\weather.nominal.arff";
        mlp.simpleWekaTrain(path);
    }

    public void simpleWekaTrain(String filepath) {
        try {
            //Reading training arff or csv file
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filepath);
            Instances train = source.getDataSet();

            train.setClassIndex(train.numAttributes() - 1);

            //MultilayerPerceptron object creat
            MultilayerPerceptron mlp = new MultilayerPerceptron();

            //Setting Parameters
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("3");

            //train the perceptron
            mlp.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
            System.out.println(eval.toSummaryString()); //Summary of Training

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}
