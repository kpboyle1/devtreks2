using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Text;

namespace DevTreks.Extensions.Algorithms
{
    /// <summary>
    ///Purpose:		ML01 algorithm
    ///Author:		www.devtreks.org
    ///Date:		2018, May
    ///References:	naive bayes machine learning
    ///adapted from McCaffrey (MSDN, February, 2013) 
    ///</summary>
    public class ML01 : MLBase
    {
        public ML01()
            : base() { }
        public ML01(int indicatorIndex, string label, string[] mathTerms,
            string[] colNames, string[] depColNames,
            string subalgorithm, int ciLevel, int iterations,
            int random, IndicatorQT1 qT1, CalculatorParameters calcParams)
            : base(indicatorIndex, label, mathTerms,
            colNames, depColNames, subalgorithm, ciLevel, iterations,
            random, qT1, calcParams)
        {

        }
        public async Task<bool> RunAlgorithmAsync(List<List<string>> trainData,
            List<List<string>> rowNames, List<List<string>> testData)
        {
            //the bool allows errors to be propagated
            bool bHasCalculations = false;
            try
            {
                //minimal data requirement is first five cols and 3 rows
                if (_colNames.Length < 5 || rowNames.Count < 3)
                {
                    IndicatorQT.ErrorMessage = "ML analysis requires at least 1 output " +
                        "variable and 1 input variable with 3 rows of test data.";
                    return bHasCalculations;
                }
                StringBuilder sResults = new StringBuilder();
                if (_subalgorithm == MATHML_SUBTYPES.subalgorithm_01.ToString())
                {
                    
                    //classify testdata and return new dataset
                    bHasCalculations = await Classify(trainData, rowNames, testData);
                    bHasCalculations = await SetMathResult(rowNames);

                    //debug first with reference dataset and show debugging messages in results
                    //sResults = await DebugClassify(trainData, rowNames, testData);
                    //put the results in MathResult
                    //bHasCalculations = await SetMathResult(rowNames, sResults);
                }

            }
            catch (Exception ex)
            {
                IndicatorQT.ErrorMessage = ex.Message;
            }
            return bHasCalculations;
        }
        
        private async Task<bool> Classify(List<List<string>> trainData,
            List<List<string>> rowNames, List<List<string>> testData)
        {
            bool bHasClassified = false;
            try
            {
                //make a new list with same matrix, to be replaced with results
                int iRowCount = Shared.GetRowCount(_iterations, testData.Count);
                int iColCount = testData[0].Count;
                if (_subalgorithm == MATHML_SUBTYPES.subalgorithm_01.ToString().ToString())
                {
                    //subalgo01 needs qtm and percent probability of qtm
                    iColCount = testData[0].Count + 2;
                }
                DataResults = CalculatorHelpers.GetList(iRowCount, iColCount);
                // trainData columns define number of rows (depcolumns.Length + 1)
                string[][] attributeValues = new string[trainData[0].Count][];
                //for each column of trainData, fill in the unique attribute names (i.e. gender = 2 unique atts)
                for (int i = 0; i < trainData[0].Count; i++)
                {
                    attributeValues[i] = Shared.GetAttributeGroups(i, trainData, this.IndicatorQT);
                }
                int[][][] jointCounts = MakeJointCounts(trainData, attributeValues);
                int[] dependentCounts = MakeDependentCounts(jointCounts, attributeValues[0].Length);
                // prevent joint counts with 0
                bool withLaplacian = true;
                //classify everything in test dataset and add result to new columns in test dataset
                //display averages in Indicator metadata
                int row = 0;
                foreach (List<string> data in testData)
                {
                    //prepare mathresults
                    for (int i = 0; i < data.Count; i++)
                    {
                        DataResults[row][i] = data[i];
                    }
                    int c = await Classify(row, attributeValues, data.ToArray(),
                        jointCounts, dependentCounts, withLaplacian, attributeValues.Length - 1);
                    for (int l = 0; l < attributeValues[0].Length; l++)
                    {
                        if (c == l)
                        {
                            int iRowLength = DataResults[row].Count;
                            DataResults[row][iRowLength - 2] = attributeValues[0][l].ToString();
                        }
                    }
                    row++;
                }
                bHasClassified = true;
            }
            catch (Exception ex)
            {
                IndicatorQT.ErrorMessage = ex.Message;
            }
            return bHasClassified;
        }
        //strictly used to debug algorithms
        private async Task<StringBuilder> DebugClassify(List<List<string>> trainData,
            List<List<string>> rowNames, List<List<string>> testData)
        {
            StringBuilder sb = new StringBuilder();
            try
            {
                // trainData columns define number of rows (depcolumns.Length + 1)
                string[][] attributeValues = new string[trainData[0].Count][];
                //for each column of trainData, fill in the unique attribute names (i.e. gender = 2 unique atts)
                for (int i = 0; i < trainData[0].Count; i++)
                {
                    attributeValues[i] = Shared.GetAttributeGroups(i, trainData);
                }
                sb.AppendLine("First 4 lines of training data are:\n");
                for (int i = 0; i < 4; ++i)
                {
                    List<string> atts = trainData[i];
                    string sData = string.Empty;
                    foreach (string att in atts)
                    {
                        sData += string.Concat(" ", att);
                    }
                    sb.AppendLine(sData);
                }
                int[][][] jointCounts = MakeJointCounts(trainData, attributeValues);
                int[] dependentCounts = MakeDependentCounts(jointCounts, attributeValues[0].Length);
                for (int j = 0; j < dependentCounts.Length; j++)
                {
                    if (attributeValues[0].Length > j)
                    {
                        string sDependent = attributeValues[0][j];
                        sb.AppendLine(string.Concat("Total for ",
                            sDependent, "s  = " + dependentCounts[j]));
                    }
                }
                ShowJointCounts(sb, jointCounts, attributeValues);

                // prevent joint counts with 0
                bool withLaplacian = true;
                sb.AppendLine("Using Naive Bayes " + (withLaplacian ? "with" : "without") + " Laplacian smoothing to classify when:");
                //classify everything in test dataset and add result to new columns in test dataset
                //display averages in Indicator metadata
                int k = 0;
                foreach (List<string> data in testData)
                {
                    string sData2 = string.Empty;
                    if (k == 0)
                    {
                        for (int i = 1; i < data.Count; i++)
                        {
                            if (i < (_depColNames.Length + 1))
                            {
                                sData2 += string.Concat(" ", _depColNames[i - 1], " = ", data[i]);
                            }
                            else
                            {
                                sData2 += string.Concat(" ", " = ", data[i]);
                            }
                        }
                        sb.AppendLine(sData2);
                        //for debugging algo
                        int c = await Classify(k, attributeValues, testData[0].ToArray(),
                            jointCounts, dependentCounts, withLaplacian, 
                            attributeValues.Length - 1, sb);
                        for (int l = 0; l < attributeValues[0].Length; l++)
                        {
                            if (c == l)
                            {
                                sb.AppendLine(string.Concat("\nThe subject is most likely to be ",
                                        attributeValues[0][l].ToString()));
                            }
                        }
                    }
                    else
                    {

                    }
                    k++;
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine(ex.Message);
            }
            return sb;
        }
        private int[][][] MakeJointCounts(List<List<string>> trainData,
            string[][] attributeValues)
        {
            // assumes binned trainData is occupation, dominance, height, sex
            // result[][][] -> [attribute][att value][sex]
            // ex: result[0][3][1] is the count of (occupation) (technology) (female), i.e., the count of technology AND female
            // note the -1 (no label or dep variable in joint counts)
            int[][][] jointCounts = new int[attributeValues.Length - 1][][];
            //first column holds labels and first row has label names
            int iLabelCount = attributeValues[0].Length;
            for (int i = 1; i < attributeValues.Length; i++)
            {
                //features start in 2nd row
                jointCounts[i - 1] = new int[attributeValues[i].Length][];
                for (int j = 0; j < attributeValues[i].Length; j++)
                {
                    //number of labels for each feature
                    jointCounts[i - 1][j] = new int[iLabelCount];
                }
            }
            for (int i = 1; i < attributeValues.Length; ++i)
            {
                for (int k = 0; k < trainData.Count; k++)
                {
                    int iLabelIndex = AttributeValueToLabelIndex(trainData[k][0], attributeValues);
                    int iFeatureIndex = 0;
                    //feature comes from each column
                    if (i < trainData[k].Count)
                    {
                        string sFeature = trainData[k][i];
                        iFeatureIndex = AttributeValueToIndex(i, sFeature, attributeValues);
                        if (jointCounts[i - 1].Length > iFeatureIndex)
                        {
                            ++jointCounts[i - 1][iFeatureIndex][iLabelIndex];
                        }
                    }
                }
            }
            return jointCounts;
        }
        private int AttributeValueToLabelIndex(string attributeValue, string[][] attributeValues)
        {
            int iAttributeIndex = -1;
            string sAttribute = string.Empty;
            //labels are in atts[0]
            for (int j = 0; j < attributeValues[0].Length; j++)
            {
                sAttribute = Shared.ConvertAttributeToLabel(attributeValue, this.IndicatorQT);
                if (sAttribute.Equals(attributeValues[0][j]))
                {
                    iAttributeIndex = j;
                    break;
                }
            }
            return iAttributeIndex;
        }
        static int AttributeValueToIndex(int firstIndex, string attributeValue, string[][] attributeValues)
        {
            int iAttributeIndex = -1;
            if (firstIndex < attributeValues.Length)
            {
                for (int j = 0; j < attributeValues[firstIndex].Length; j++)
                {
                    if (attributeValue.Equals(attributeValues[firstIndex][j]))
                    {
                        iAttributeIndex = j;
                        break;
                    }
                }
            }
            return iAttributeIndex;
        }

        static void ShowJointCounts(StringBuilder sb, int[][][] jointCounts, string[][] attributeValues)
        {
            string sFeature = string.Empty;
            for (int k = 0; k < attributeValues[0].Length; ++k)
            {
                for (int i = 0; i < jointCounts.Length; ++i)
                    for (int j = 0; j < jointCounts[i].Length; ++j)
                        if (((i+1) < attributeValues.Length) 
                            && (j < attributeValues[i+1].Length))
                        {
                            sb.AppendLine(string.Concat(attributeValues[i+1][j].PadRight(15),
                                "& ", attributeValues[0][k].PadRight(6),
                                " = ", jointCounts[i][j][k]));
                        }
                // separate sexes
                sb.AppendLine(""); 
            }
        }

        static int[] MakeDependentCounts(int[][][] jointCounts, int numDependents)
        {
            int[] result = new int[numDependents];
            // male then female
            for (int k = 0; k < numDependents; ++k)
                // scanning attribute 0 = occupation. could use any attribute
                for (int j = 0; j < jointCounts[0].Length; ++j) 
                    result[k] += jointCounts[0][j][k];

            return result;
        }

        private async Task<int> Classify(int row, string[][] attributeValues, 
            string[] featuresToTest, int[][][] jointCounts, int[] dependentCounts, 
            bool withSmoothing, int xClasses, StringBuilder sb = null)
        {
            double partProbLabel = 0;
            double totalProbability = 0;
            //1st row holds the dependent vars classifations
            double[] partProbLabels = new double[attributeValues[0].Length];
            for (int k = 0; k < attributeValues[0].Length; ++k)
            {
                partProbLabel = await PartialProbability(attributeValues[0][k], attributeValues,
                    featuresToTest, jointCounts, dependentCounts, 
                    withSmoothing, xClasses);
                totalProbability += partProbLabel;
                partProbLabels.SetValue(partProbLabel, k);
            }
            double fullProbability = 0;
            double highestProbability = 0;
            int iLabelIndex = -1;
            for (int k = 0; k < attributeValues[0].Length; ++k)
            {
                if (sb != null)
                {
                    sb.AppendLine(string.Concat("Partial prob of ",
                        attributeValues[0][k], "= ", partProbLabels[k].ToString("F6")));
                }
                if (partProbLabels[k] > highestProbability)
                {
                    highestProbability = partProbLabels[k];
                    iLabelIndex = k;
                }
                fullProbability = partProbLabels[k] / totalProbability;
                if (sb != null)
                {
                    sb.AppendLine(string.Concat("Probability of ",
                        attributeValues[0][k], "= ", fullProbability.ToString("F4")));
                }
                else
                {
                    int iRowLength = DataResults[row].Count;
                    DataResults[row][iRowLength - 1] = fullProbability.ToString("F4");
                }
            }
            //return label that has highest probability
            return iLabelIndex;
        }
        
        private static async Task<double> PartialProbability(string label, string[][] attributeValues, string[] featuresToTest, 
            int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
        {
            int iLabelIndex = AttributeValueToIndex(0, label, attributeValues);
            int iFeatureIndex = 0;
            //test data includes dependent var with na or none
            int[] iFeatureIndexes = new int[featuresToTest.Length - 1];
            int i = 0;
            foreach (string feature in featuresToTest)
            {
                if (i > 0)
                {
                    iFeatureIndex = AttributeValueToIndex(i, feature, attributeValues);
                    if ((i -1) < iFeatureIndexes.Length)
                    {
                        iFeatureIndexes[i - 1] = iFeatureIndex;
                    }
                }
                i++;
            }
            int totalCases = 0;
            for (i = 0; i < dependentCounts.Length; i++)
            {
                totalCases += dependentCounts[i];
            }
            int totalToUse = 0;
            i = 0;
            foreach (string attLabel in attributeValues[0])
            {
                if (attLabel.Equals(label))
                {
                    if (i < dependentCounts.Length)
                    {
                        totalToUse = dependentCounts[i];
                        break;
                    }
                }
                i++;
            }
            double[] ps = new double[iFeatureIndexes.Length + 1];
            // prob of either male or female
            double p0 = (totalToUse * 1.0) / (totalCases);
            ps.SetValue(p0, 0);
            double p1 = 0.0;
            if (withSmoothing == false)
            {
                i = 0;
                foreach (int featureIndex in iFeatureIndexes)
                {
                    // conditional prob of label given the features
                    p1 = (jointCounts[i][featureIndex][iLabelIndex] * 1.0) / totalToUse;
                    if (ps.Length > (i + 1))
                    {
                        ps.SetValue(p1, i + 1);
                    }
                    i++;
                }    
            }
            else if (withSmoothing == true) 
            {
                // Laplacian smoothing to avoid 0-count joint probabilities
                i = 0;
                foreach (int featureIndex in iFeatureIndexes)
                {
                    // conditional prob of label given the features
                    p1 = (jointCounts[i][featureIndex][iLabelIndex] + 1) / ((totalToUse + xClasses) * 1.0);
                    if (ps.Length > (i + 1))
                    {
                        ps.SetValue(p1, i + 1);
                    }
                    i++;
                }
            }
            double dbTotalProbability = 0;
            foreach (double p in ps)
            {
                dbTotalProbability += Math.Log(p);
            }
            dbTotalProbability = Math.Exp(dbTotalProbability);
            return dbTotalProbability;
        }
        
        static int AnalyzeJointCounts(int[][][] jointCounts)
        {
            // check for any joint-counts that are 0 which could blow up Naive Bayes
            int zeroCounts = 0;
            // attribute
            for (int i = 0; i < jointCounts.Length; ++i)
                // value
                for (int j = 0; j < jointCounts[i].Length; ++j)
                    // label
                    for (int k = 0; k < jointCounts[i][j].Length; ++k) 
                        if (jointCounts[i][j][k] == 0)
                            ++zeroCounts;
            return zeroCounts;
        }
        private async Task<bool> SetMathResult(List<List<string>> rowNames, StringBuilder sb = null)
        {
            bool bHasSet = false;
            if (sb == null)
            {
                //add the data to a string builder
                sb = new StringBuilder();
                if (_subalgorithm == MATHML_SUBTYPES.subalgorithm_01.ToString())
                {
                    sb.AppendLine("ml results");
                    string[] newColNames = new string[_colNames.Length + 2];
                    for (int i = 0; i < _colNames.Length; i++)
                    {
                        newColNames[i] = _colNames[i];
                    }
                    //new cols changed by algo
                    newColNames[_colNames.Length] = "label";
                    newColNames[_colNames.Length + 1] = "probability";
                    _colNames = newColNames;
                    CalculatorHelpers.SetIndMathResult(sb, _colNames, rowNames, DataResults);
                }

            }
            if (this.IndicatorQT.MathResult.ToLower().StartsWith("http"))
            {
                bool bHasSaved = await CalculatorHelpers.SaveTextInURI(
                    Params.ExtensionDocToCalcURI, sb.ToString(), this.IndicatorQT.MathResult);
                if (!string.IsNullOrEmpty(Params.ExtensionDocToCalcURI.ErrorMessage))
                {
                    this.IndicatorQT.MathResult += Params.ExtensionDocToCalcURI.ErrorMessage;
                    //done with errormsg
                    Params.ExtensionDocToCalcURI.ErrorMessage = string.Empty;
                }
                else
                {
                    bHasSet = true;
                }
            }
            else
            {
                this.IndicatorQT.MathResult = sb.ToString();
                bHasSet = true;
            }
            return bHasSet;
        }
    }
}
