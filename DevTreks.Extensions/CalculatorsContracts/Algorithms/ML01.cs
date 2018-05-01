using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Text;

namespace DevTreks.Extensions.Algorithms
{
    /// <summary>
    ///Purpose:		ML01 algorithm
    ///Author:		www.devtreks.org
    ///Date:		2018, April
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
        
        public async Task<bool> RunAlgorithmAsync(List<List<string>> data,
                List<List<string>> rowNames, List<List<string>> data2)
        {
            //the bool allows errors to be propagated
            bool bHasCalculations = false;
            try
            {
                string[] args = new string[] { };
                StringBuilder sResults = GetCalculation(data, rowNames, data2);
                //put the results in MathResult
                bHasCalculations = await SetMathResult(rowNames, sResults);

            }
            catch (Exception ex)
            {
                IndicatorQT.ErrorMessage = ex.Message;
            }
            return bHasCalculations;
        }
        private static StringBuilder GetCalculation(List<List<string>> data1,
                List<List<string>> rowNames, List<List<string>> data2)
        {
            StringBuilder sb = new StringBuilder();
            try
            {
                // data1 columns define number of rows (depcolumns.Length + 1)
                string[][] attributeValues = new string[data1[0].Count][];
                //for each column of data, fill in the unique attribute names (i.e. gender = 2 unique atts)
                for (int i = 0; i < data1[0].Count; i++)
                {
                    attributeValues[i] = Shared.GetAttributeGroups(i, data1);
                }
                sb.AppendLine("First 4 lines of training data are:\n");

                for (int i = 0; i < 4; ++i)
                {
                    List<string> atts = data1[i];
                    string sData = string.Empty;
                    foreach (string att in atts)
                    {
                        sData += string.Concat(" ", att);
                    }
                    sb.AppendLine(sData);
                }
                sb.AppendLine("\n");
                int[][][] jointCounts = MakeJointCounts(data1, attributeValues);
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
                sb.AppendLine("");
                ShowJointCounts(sb, jointCounts, attributeValues);

                // prevent joint counts with 0
                bool withLaplacian = true;
                sb.AppendLine("Using Naive Bayes " + (withLaplacian ? "with" : "without") + " Laplacian smoothing to classify when:");
                //for (int i = 0; i < 4; ++i)
                //{
                    List<string> att2s = data2[0];
                    string sData2 = string.Empty;
                    foreach (string att in att2s)
                    {
                        sData2 += string.Concat(" ", att);
                    }
                    sb.AppendLine(sData2);
                //}
                sb.AppendLine("\n");
                int c = Classify(sb, attributeValues, data2[0].ToArray(),
                    jointCounts, dependentCounts, withLaplacian, attributeValues.Length - 1);
                for (int l = 0; l < attributeValues[0].Length; l++)
                {
                    if (c == l)
                    {
                        sb.AppendLine(string.Concat("\nThe household is most likely to be ",
                                attributeValues[0][l].ToString()));
                    }
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine(ex.Message);
            }
            return sb;
        }
        
        static int[][][] MakeJointCounts(List<List<string>> data1,
            string[][] attributeValues)
        {
            // assumes binned data is occupation, dominance, height, sex
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
                for (int k = 0; k < data1.Count; k++)
                {
                    int iLabelIndex = AttributeValueToIndex(data1[k][0], attributeValues);
                    int iFeatureIndex = 0;
                    //feature comes from each column
                    if (i < data1[k].Count)
                    {
                        string sFeature = data1[k][i];
                        iFeatureIndex = AttributeValueToIndex(sFeature, attributeValues);
                        if (jointCounts[i - 1].Length > iFeatureIndex)
                        {
                            ++jointCounts[i - 1][iFeatureIndex][iLabelIndex];
                        }
                    }
                }
            }
            return jointCounts;
        }

        static int AttributeValueToIndex(string attributeValue, string[][] attributeValues)
        {
            int iAttributeIndex = -1;
            for (int i = 0; i < attributeValues.Length; ++i)
            {
                for (int j = 0; j < attributeValues[i].Length; j++)
                {
                    if (attributeValue.Equals(attributeValues[i][j]))
                    {
                        iAttributeIndex = j;
                        break;
                    }
                }
                if (iAttributeIndex != -1)
                {
                    break;
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

        static int Classify(StringBuilder sb, string[][] attributeValues, 
            string[] featuresToTest, int[][][] jointCounts, int[] dependentCounts, 
            bool withSmoothing, int xClasses)
        {
            double partProbLabel = 0;
            double totalProbability = 0;
            double[] partProbLabels = new double[attributeValues[0].Length];
            for (int k = 0; k < attributeValues[0].Length; ++k)
            {
                partProbLabel = PartialProbability(attributeValues,
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
                sb.AppendLine(string.Concat("Partial prob of ", 
                    attributeValues[0][k], "= ", partProbLabels[k].ToString("F6")));
                if (partProbLabels[k] > highestProbability)
                {
                    highestProbability = partProbLabels[k];
                    iLabelIndex = k;
                }
                fullProbability = partProbLabels[k] / totalProbability;
                sb.AppendLine(string.Concat("Probability of ",
                    attributeValues[0][k], "= ", fullProbability.ToString("F4")));
            }
            //return label that has highest probability
            return iLabelIndex;
        }

        static double PartialProbability(string[][] attributeValues, string[] featuresToTest, 
            int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
        {
            string label = string.Empty;
            int iLabelIndex = -1;
            int iFeatureIndex = 0;
            int iFCount = featuresToTest.Length - 1;
            int[] iFeatureIndexes = new int[iFCount];
            int i = 0;
            foreach (string feature in featuresToTest)
            {
                if (i == 0)
                {
                    label = feature;
                    iLabelIndex = AttributeValueToIndex(label, attributeValues);
                }
                else
                {
                    iFeatureIndex = AttributeValueToIndex(feature, attributeValues);
                    if ((i-1) < iFeatureIndexes.Length)
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
                    // conditional prob of male (or female, depending on sex parameter) given the occupation
                    p1 = (jointCounts[i][featureIndex][iLabelIndex] * 1.0) / totalToUse;
                    if (ps.Length > (i + 1))
                    {
                        ps.SetValue(p1, i + 1);
                    }
                    i++;
                }    
            }
            else if (withSmoothing == true) // Laplacian smoothing to avoid 0-count joint probabilities
            {
                i = 0;
                foreach (int featureIndex in iFeatureIndexes)
                {
                    // conditional prob of male (or female, depending on sex parameter) given the occupation
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
                dbTotalProbability += Math.Exp(Math.Log(p));
            }
            //return Math.Exp(Math.Log(p0) + Math.Log(p1) + Math.Log(p2) + Math.Log(p3));
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
        private async Task<bool> SetMathResult(List<List<string>> rowNames, StringBuilder sb)
        {
            bool bHasSet = false;
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



        //original McCaffrey code
        // 25 gives a nice demo
        //private static Random _ran = new Random(25);
        //static StringBuilder GetCalculation(string[] args)
        //{
        //    StringBuilder sb = new StringBuilder();
        //    try
        //    {
        //        string[] attributes = new string[] { "occupation", "dominance", "height", "sex" };
        //        // could scan values from raw data
        //        string[][] attributeValues = new string[attributes.Length][];
        //        attributeValues[0] = new string[] { "administrative", "construction", "education", "technology" };
        //        attributeValues[1] = new string[] { "left", "right" };
        //        attributeValues[2] = new string[] { "short", "medium", "tall" };
        //        attributeValues[3] = new string[] { "male", "female" };
        //        // there may be several numeric variables
        //        double[][] numericAttributeBorders = new double[1][];
        //        // height range: [57.0 to 78.0]
        //        numericAttributeBorders[0] = new double[] { 64.0, 71.0 };

        //        string[] data = MakeData(40);

        //        sb.AppendLine("First 4 lines of training data are:\n");

        //        for (int i = 0; i < 4; ++i)
        //            sb.AppendLine(data[i]);
        //        sb.AppendLine("\n");
        //        // convert numeric heights to categories
        //        string[] binnedData = BinData(data, attributeValues, numericAttributeBorders);

        //        sb.AppendLine("First 4 lines of binned training data are:\n");
        //        for (int i = 0; i < 4; ++i)
        //            sb.AppendLine(binnedData[i]);
        //        sb.AppendLine("\n");

        //        int[][][] jointCounts = MakeJointCounts(binnedData, attributes, attributeValues);
        //        int[] dependentCounts = MakeDependentCounts(jointCounts, 2);
        //        sb.AppendLine("Total male = " + dependentCounts[0]);
        //        sb.AppendLine("Total female = " + dependentCounts[1]);
        //        sb.AppendLine("");

        //        ShowJointCounts(sb, jointCounts, attributeValues);

        //        // classify the sex of a person whose occupation is education, is right-handed, and is tall
        //        string occupation = "education";
        //        string dominance = "right";
        //        string height = "tall";
        //        // prevent joint counts with 0
        //        bool withLaplacian = true;

        //        sb.AppendLine("Using Naive Bayes " + (withLaplacian ? "with" : "without") + " Laplacian smoothing to classify when:");
        //        sb.AppendLine(" occupation = " + occupation);
        //        sb.AppendLine(" dominance = " + dominance);
        //        sb.AppendLine(" height = " + height);
        //        sb.AppendLine("");

        //        int c = Classify(sb, occupation, dominance, height, jointCounts, dependentCounts, withLaplacian, 3);
        //        if (c == 0)
        //            sb.AppendLine("\nThe household is most likely headed by a male.");
        //        else if (c == 1)
        //            sb.AppendLine("\nThe household is most likely headed by a female.");
        //    }
        //    catch (Exception ex)
        //    {
        //        sb.AppendLine(ex.Message);
        //    }
        //    return sb;
        //}

        //static string[] MakeData(int numRows) // make dummy data
        //{
        //    string[] result = new string[numRows];
        //    for (int i = 0; i < numRows; ++i)
        //    {
        //        string sex = MakeSex();
        //        string occupation = MakeOccupation(sex);
        //        string dominance = MakeDominance(sex);
        //        string height = MakeHeight(sex);
        //        string s = occupation + "," + dominance + "," + height + "," + sex;
        //        result[i] = s;
        //    }
        //    return result;
        //}

        //static string MakeSex()
        //{
        //    int r = _ran.Next(0, 19);
        //    if (r >= 0 && r <= 11) return "male"; // 60%
        //    else if (r >= 12 && r <= 19) return "female"; // 40%
        //    return "error";
        //}

        //static string MakeDominance(string sex)
        //{
        //    double p = _ran.NextDouble();
        //    if (sex == "male")
        //    {
        //        if (p < 0.33) return "left"; else return "right";
        //    }
        //    else if (sex == "female")
        //    {
        //        if (p < 0.20) return "left"; else return "right";
        //    }
        //    return "error";
        //}

        //static string MakeOccupation(string sex)
        //{
        //    int r = _ran.Next(0, 20);
        //    if (sex == "male")
        //    {
        //        if (r == 0) return "administrative"; // 5%
        //        else if (r >= 1 && r <= 6) return "construction"; // 30%
        //        else if (r >= 7 && r <= 9) return "education"; // 15%
        //        else if (r >= 10 && r <= 19) return "technology"; // 50%
        //    }
        //    else if (sex == "female")
        //    {
        //        if (r >= 0 & r <= 9) return "administrative"; // 50%
        //        else if (r == 10) return "construction"; // 5%
        //        else if (r >= 11 & r <= 15) return "education";  // 25%
        //        else if (r >= 16 && r <= 19) return "technology"; // 20%
        //    }
        //    return "error";
        //}

        //static string MakeHeight(string sex)
        //{
        //    int bucket = 0;  // height bucket: 0 = short, 1 = medium, 2 = tall
        //    double p = _ran.NextDouble();
        //    if (p < 0.1587) bucket = 0;
        //    else if (p > 0.8413) bucket = 2;
        //    else bucket = 1; // p = (2 * 0.3413) = 0.6826

        //    double hi = 0.0;
        //    double lo = 0.0;

        //    if (sex == "male")
        //    {
        //        if (bucket == 0) { lo = 60.0; hi = 66.0; }
        //        else if (bucket == 1) { lo = 66.0; hi = 72.0; }
        //        else if (bucket == 2) { lo = 72.0; hi = 78.0; }
        //    }
        //    else if (sex == "female")
        //    {
        //        if (bucket == 0) { lo = 57.0; hi = 63.0; }
        //        else if (bucket == 1) { lo = 63.0; hi = 69.0; }
        //        else if (bucket == 2) { lo = 69.0; hi = 75.0; }
        //    }

        //    double resultAsDouble = (hi - lo) * _ran.NextDouble() + lo;
        //    return resultAsDouble.ToString("F1");
        //}

        //static string[] BinData(string[] data, string[][] attributeValues, double[][] numericAttributeBorders)
        //{
        //    // convert numeric height to "short", "medium", or "tall". assumes data is occupation,dominance,height,sex
        //    string[] result = new string[data.Length];
        //    string[] tokens;
        //    double heightAsDouble;
        //    string heightAsBinnedString;

        //    for (int i = 0; i < data.Length; ++i)
        //    {
        //        tokens = data[i].Split(',');
        //        heightAsDouble = double.Parse(tokens[2]);
        //        if (heightAsDouble <= numericAttributeBorders[0][0]) // short
        //            heightAsBinnedString = attributeValues[2][0];
        //        else if (heightAsDouble >= numericAttributeBorders[0][1]) // tall
        //            heightAsBinnedString = attributeValues[2][2];
        //        else
        //            heightAsBinnedString = attributeValues[2][1]; // medium

        //        string s = tokens[0] + "," + tokens[1] + "," + heightAsBinnedString + "," + tokens[3];
        //        result[i] = s;
        //    }
        //    return result;
        //}

        //static int[][][] MakeJointCounts(string[] binnedData, string[] attributes, string[][] attributeValues)
        //{
        //    // assumes binned data is occupation, dominance, height, sex
        //    // result[][][] -> [attribute][att value][sex]
        //    // ex: result[0][3][1] is the count of (occupation) (technology) (female), i.e., the count of technology AND female
        //    // note the -1 (no sex)
        //    int[][][] jointCounts = new int[attributes.Length - 1][][];
        //    // 4 occupations
        //    jointCounts[0] = new int[4][];
        //    // 2 dominances
        //    jointCounts[1] = new int[2][];
        //    // 3 heights
        //    jointCounts[2] = new int[3][];

        //    jointCounts[0][0] = new int[2]; // 2 sexes for administrative
        //    jointCounts[0][1] = new int[2]; // construction
        //    jointCounts[0][2] = new int[2]; // education
        //    jointCounts[0][3] = new int[2]; // tedchnology

        //    jointCounts[1][0] = new int[2]; // left
        //    jointCounts[1][1] = new int[2]; // right

        //    jointCounts[2][0] = new int[2]; // short
        //    jointCounts[2][1] = new int[2]; // medium
        //    jointCounts[2][2] = new int[2]; // tall

        //    for (int i = 0; i < binnedData.Length; ++i)
        //    {
        //        string[] tokens = binnedData[i].Split(',');

        //        int occupationIndex = AttributeValueToIndex(0, tokens[0]);
        //        int dominanceIndex = AttributeValueToIndex(1, tokens[1]);
        //        int heightIndex = AttributeValueToIndex(2, tokens[2]);
        //        int sexIndex = AttributeValueToIndex(3, tokens[3]);
        //        // occupation and sex count
        //        ++jointCounts[0][occupationIndex][sexIndex];
        //        ++jointCounts[1][dominanceIndex][sexIndex];
        //        ++jointCounts[2][heightIndex][sexIndex];
        //    }

        //    return jointCounts;
        //}

        //static int AttributeValueToIndex(int attribute, string attributeValue)
        //{
        //    // we could do this programmatically (maybe with a Dictionary) but a crude approach is more clear
        //    if (attribute == 0)
        //    {
        //        if (attributeValue == "administrative") return 0;
        //        else if (attributeValue == "construction") return 1;
        //        else if (attributeValue == "education") return 2;
        //        else if (attributeValue == "technology") return 3;
        //    }
        //    else if (attribute == 1)
        //    {
        //        if (attributeValue == "left") return 0;
        //        else if (attributeValue == "right") return 1;
        //    }
        //    else if (attribute == 2)
        //    {
        //        if (attributeValue == "short") return 0;
        //        else if (attributeValue == "medium") return 1;
        //        else if (attributeValue == "tall") return 2;
        //    }
        //    else if (attribute == 3)
        //    {
        //        if (attributeValue == "male") return 0;
        //        else if (attributeValue == "female") return 1;
        //    }
        //    return -1; // error
        //}

        //static void ShowJointCounts(StringBuilder sb, int[][][] jointCounts, string[][] attributeValues)
        //{
        //    for (int k = 0; k < 2; ++k)
        //    {
        //        for (int i = 0; i < jointCounts.Length; ++i)
        //            for (int j = 0; j < jointCounts[i].Length; ++j)
        //                sb.AppendLine(attributeValues[i][j].PadRight(15) + "& " + attributeValues[3][k].PadRight(6) + " = " + jointCounts[i][j][k]);
        //        // separate sexes
        //        sb.AppendLine("");
        //    }
        //}

        //static int[] MakeDependentCounts(int[][][] jointCounts, int numDependents)
        //{
        //    int[] result = new int[numDependents];
        //    // male then female
        //    for (int k = 0; k < numDependents; ++k)
        //        // scanning attribute 0 = occupation. could use any attribute
        //        for (int j = 0; j < jointCounts[0].Length; ++j)
        //            result[k] += jointCounts[0][j][k];

        //    return result;
        //}

        //static int Classify(StringBuilder sb, string occupation, string dominance, string height, int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
        //{
        //    double partProbMale = PartialProbability("male", occupation, dominance, height, jointCounts, dependentCounts, withSmoothing, xClasses);
        //    double partProbFemale = PartialProbability("female", occupation, dominance, height, jointCounts, dependentCounts, withSmoothing, xClasses);
        //    double evidence = partProbMale + partProbFemale;
        //    double probMale = partProbMale / evidence;
        //    double probFemale = partProbFemale / evidence;

        //    //sb.AppendLine("Partial prob of male   = " + partProbMale.ToString("F6"));
        //    //sb.AppendLine("Partial prob of female = " + partProbFemale.ToString("F6"));

        //    sb.AppendLine("Probability of male headed household  = " + probMale.ToString("F4"));
        //    sb.AppendLine("Probability of female headed household = " + probFemale.ToString("F4"));
        //    // could use a threshold
        //    if (probMale > probFemale)
        //        return 0;
        //    else
        //        return 1;
        //}

        //static double PartialProbability(string sex, string occupation, string dominance, string height, int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
        //{
        //    int sexIndex = AttributeValueToIndex(3, sex);

        //    int occupationIndex = AttributeValueToIndex(0, occupation);
        //    int dominanceIndex = AttributeValueToIndex(1, dominance);
        //    int heightIndex = AttributeValueToIndex(2, height);

        //    int totalMale = dependentCounts[0];
        //    int totalFemale = dependentCounts[1];
        //    int totalCases = totalMale + totalFemale;

        //    int totalToUse = 0;
        //    if (sex == "male") totalToUse = totalMale;
        //    else if (sex == "female") totalToUse = totalFemale;
        //    // prob of either male or female
        //    double p0 = (totalToUse * 1.0) / (totalCases);
        //    double p1 = 0.0;
        //    double p2 = 0.0;
        //    double p3 = 0.0;

        //    if (withSmoothing == false)
        //    {
        //        // conditional prob of male (or female, depending on sex parameter) given the occupation
        //        p1 = (jointCounts[0][occupationIndex][sexIndex] * 1.0) / totalToUse;
        //        // conditional prob of the specified sex, given the specified domnance
        //        p2 = (jointCounts[1][dominanceIndex][sexIndex] * 1.0) / totalToUse;
        //        // condition prob given specified height
        //        p3 = (jointCounts[2][heightIndex][sexIndex] * 1.0) / totalToUse;
        //    }
        //    else if (withSmoothing == true) // Laplacian smoothing to avoid 0-count joint probabilities
        //    {
        //        // add 1 to count in numerator, add number x classes in denominator
        //        p1 = (jointCounts[0][occupationIndex][sexIndex] + 1) / ((totalToUse + xClasses) * 1.0);
        //        // conditional prob of the specified sex, given the specified domnance
        //        p2 = (jointCounts[1][dominanceIndex][sexIndex] + 1) / ((totalToUse + xClasses) * 1.0);
        //        p3 = (jointCounts[2][heightIndex][sexIndex] + 1) / ((totalToUse + xClasses) * 1.0);
        //    }

        //    //return p0 * p1 * p2 * p3; // risky if any very small values
        //    return Math.Exp(Math.Log(p0) + Math.Log(p1) + Math.Log(p2) + Math.Log(p3));
        //}

        //static int AnalyzeJointCounts(int[][][] jointCounts)
        //{
        //    // check for any joint-counts that are 0 which could blow up Naive Bayes
        //    int zeroCounts = 0;
        //    // attribute
        //    for (int i = 0; i < jointCounts.Length; ++i)
        //        // value
        //        for (int j = 0; j < jointCounts[i].Length; ++j)
        //            // sex
        //            for (int k = 0; k < jointCounts[i][j].Length; ++k)
        //                if (jointCounts[i][j][k] == 0)
        //                    ++zeroCounts;
        //    return zeroCounts;
        //}
        //private async Task<bool> SetMathResult(List<List<string>> rowNames, StringBuilder sb)
        //{
        //    bool bHasSet = false;
        //    if (this.IndicatorQT.MathResult.ToLower().StartsWith("http"))
        //    {
        //        bool bHasSaved = await CalculatorHelpers.SaveTextInURI(
        //            Params.ExtensionDocToCalcURI, sb.ToString(), this.IndicatorQT.MathResult);
        //        if (!string.IsNullOrEmpty(Params.ExtensionDocToCalcURI.ErrorMessage))
        //        {
        //            this.IndicatorQT.MathResult += Params.ExtensionDocToCalcURI.ErrorMessage;
        //            //done with errormsg
        //            Params.ExtensionDocToCalcURI.ErrorMessage = string.Empty;
        //        }
        //        else
        //        {
        //            bHasSet = true;
        //        }
        //    }
        //    else
        //    {
        //        this.IndicatorQT.MathResult = sb.ToString();
        //        bHasSet = true;
        //    }
        //    return bHasSet;
        //}
    }
}
