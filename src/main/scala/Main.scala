import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Imputer, Normalizer, OneHotEncoder, StringIndexer, UnivariateFeatureSelector, VectorAssembler}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GeneralizedLinearRegression, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, to_date, udf, year}
import org.apache.spark.sql.types.IntegerType




object Main {
  private val replaceNullWithUnknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow" || x == "None" || x == "" || x == " ") res = "unknown"
    else res = x
    res
  }).asNondeterministic()

  private val replaceNAWithNull = udf((x: String) => {
    var res = new String
    if (x == "NA") res = null
    else res = x
    res
  }).asNondeterministic()

  private val replaceTimeWithDayPart = udf((x: Integer) => {
    var res = new String
    if(x > 0 && x < 500) res = "lateNight"
    if(x >= 500 && x < 800) res = "earlyMorning"
    if(x >= 800 && x < 1200) res = "lateMorning"
    if(x >= 1200 && x < 1400) res = "earlyAfternoon"
    if(x >= 1400 && x < 1700) res = "lateAfternoon"
    if(x >= 1700 && x < 1900) res = "earlyEvening"
    if(x >= 1900 && x < 2100) res = "lateEvening"
    if(x >= 2100 && x <= 2400) res = "earlyNight"
    res
  }).asNondeterministic()

  private val replaceYMDWithFlightDate = udf((year: Integer, month: Integer, day: Integer) => {
    var res = month.toString + "/" + day.toString + "/" + year.toString
    res
  }).asNondeterministic()

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Big Data: Spark Practical Work")
      .master("local[12]")
      .config("spark.driver.memory","24G")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

//    val schema = new StructType()
//      .add("Year", IntegerType, true)
//      .add("Month", IntegerType, true)
//      .add("DayofMonth", IntegerType, true)
//      .add("DayOfWeek", IntegerType, true)
//      .add("DepTime", IntegerType, true)
//      .add("CRSDepTime", IntegerType, true)
//      .add("ArrTime", IntegerType, true)
//      .add("CRSArrTime", IntegerType, true)
//      .add("UniqueCarrier", StringType, true)
//      .add("FlightNum", IntegerType, true)
//      .add("TailNum", StringType, true)
//      .add("ActualElapsedTime", IntegerType, true)
//      .add("CRSElapsedTime", IntegerType, true)
//      .add("AirTime", IntegerType, true)
//      .add("ArrDelay", IntegerType, true)
//      .add("DepDelay", IntegerType, true)
//      .add("Origin", StringType, true)
//      .add("Dest", StringType, true)
//      .add("Distance", IntegerType, true)
//      .add("TaxiIn", IntegerType, true)
//      .add("TaxiOut", IntegerType, true)
//      .add("Cancelled", IntegerType, true)
//      .add("CancellationCode", StringType, true)
//      .add("Diverted", IntegerType, true)
//      .add("CarrierDelay", IntegerType, true)
//      .add("WeatherDelay", IntegerType, true)
//      .add("NASDelay", IntegerType, true)
//      .add("SecurityDelay", IntegerType, true)
//      .add("LateAircraftDelay", IntegerType, true)
//
//    val schema2 = new StructType()
//      .add("tailNum", StringType, true)
//      .add("type", StringType, true)
//      .add("manufacturer", StringType, true)
//      .add("issue_date", StringType, true)
//      .add("model", StringType, true)
//      .add("status", StringType, true)
//      .add("aircraft_type", StringType, true)
//      .add("engine_type", StringType, true)
//      .add("year", IntegerType, true)

    println()
    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println("---------------------------------------------------------------------- DATA LOADING ------------------------------------------------------------------------------------")
    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println()


    // We read the input data
    var df = spark.read.option("header", value = "true").csv("src/main/resources/2000.csv")
    //df = df.union(spark.read.option("header", value = "true").csv("src/main/resources/2007.csv"))
    var dfPlane = spark.read.option("header", value = "true").csv("src/main/resources/plane-data.csv")


    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println("---------------------------------------------------------------- DATA PREPROCESSING & FEATURE SELECTION ----------------------------------------------------------------")
    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println()

    // We delete the forbidden columns
    println("--------------------------------- We delete the forbidden columns -----------------------------------------------")
    val columnsToDrop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    df = df.drop(columnsToDrop:_*)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete the null values of the target variable as we are not going to use that values
    println("--------------------------------- We delete the null values of \"ArrDelay\" -----------------------------------------------")
    df = df.filter("ArrDelay is NOT NULL AND ArrDelay NOT LIKE 'NA'")
    println("--------------------------------- Done -----------------------------------------------")
    println()
    println("--------------------------------- Target variable -----------------------------------------------")
    println("'ArrDelay'")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We filter the columns that can not possess negative values
    println("--------------------------------- We filter the columns that can not possess negative values -----------------------------------------------")
    df = df.filter("Year > 0")
    df = df.filter("Month > 0")
    df = df.filter("DayofMonth > 0")
    df = df.filter("DayOfWeek > 0")
    df = df.filter("DepTime > 0")
    df = df.filter("CRSArrTime > 0")
    df = df.filter("Distance > 0")
    df = df.filter("TaxiOut > 0")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal
    println("--------------------------------- We delete all the rows that contain cancelled flights -----------------------------------------------")
    df = df.filter("Cancelled == 0")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Therefore, we eliminate the "CancellationCode" and "Cancelled" columns
    println("--------------------------------- We eliminate the \"CancellationCode\" and \"Cancelled\" columns -----------------------------------------------")
    df = df.drop("Cancelled", "CancellationCode")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete the "CRSDepTime" column given that we already have enough information with the "DepTime" and "DepDelay" columns
    println("--------------------------------- We delete the \"CRSDepTime\" column -----------------------------------------------")
    df = df.drop("CRSDepTime")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete the "CRSElapsedTime" column since this variable seems to give the same information as the "Distance" column (higher distance, higher estimated time and vice versa)
    println("--------------------------------- We delete the \"CRSElapsedTime\" column -----------------------------------------------")
    df = df.drop("CRSElapsedTime")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We clean the "issue_date" column since it is going to be used later
    // By doing this, we also delete the status column since it does not provide more useful information
    println("--------------------------------- We clean the \"issue_date\" column -----------------------------------------------")
    dfPlane = dfPlane.filter("issue_date is NOT NULL AND issue_date NOT LIKE 'None'")
    dfPlane = dfPlane.drop("status")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete the plane tailnumbers that do not have any data from plane-data dataset
    println("--------------------------------- We delete the plane tailnumbers that do not have any data from plane-data dataset -----------------------------------------------")
    dfPlane = dfPlane.filter("type is NOT NULL AND manufacturer is NOT NULL AND model is NOT NULL AND aircraft_type is NOT NULL AND engine_type is NOT NULL AND year is NOT NULL")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Renaming column "year" to "year_produced" in dfPlane dataset and eliminating it since we are not going to use it
    println("--------------------------------- Deleting column \"year\" in dfPlane dataset -----------------------------------------------")
    dfPlane = dfPlane.withColumnRenamed("year","year_introduced")
    dfPlane = dfPlane.drop("year_introduced")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Join of the two datasets
    println("--------------------------------- Joining both datasets -----------------------------------------------")
    df = df.join(dfPlane, "tailNum")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We check for NA values in the each column of the dataset and set them to null for the imputers to do their work
    println("--------------------------------- Checking for NA values in the dataset to set them to null -----------------------------------------------")
    for (i <- 0 until df.columns.drop(df.columns.indexOf("ArrDelay")).length) {
      val column = df.columns(i)
      df = df.withColumn(column, replaceNAWithNull(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Numerical columns for "mean" imputer and "most frequent" imputer
    val numColsMean = Array("DepTime","CRSArrTime","DepDelay","Distance","TaxiOut")
    val numColsMf = Array("FlightNum", "Year","Month","DayofMonth","DayOfWeek")


    // We cast to Integer every column in order to be able to use the imputer
    println("--------------------------------- We cast to Integer every column in order to be able to use the imputer -----------------------------------------------")
    for (i <- 0 until df.columns.length){
      val colName = df.columns(i)
      if (numColsMean.contains(colName) || numColsMf.contains(colName) || colName == "ArrDelay")
        df = df.withColumn(colName,col(colName).cast(IntegerType))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "most frequent" imputer for the "Year", "Month", "DayOfMonth" and "DayOfWeek" columns
    println("--------------------------------- We apply the \"most frequent\" imputer for the \"Year\",\"Month\",\"DayofMonth\" and \"DayOfWeek\" columns -----------------------------------------------")
    val imputer = new Imputer()
      .setInputCols(numColsMf)
      .setOutputCols(numColsMf)
      .setStrategy("mode")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "mean" imputer for the rest of the numerical columns
    println("--------------------------------- We apply the \"mean\" imputer for the \"DepTime\",\"CRSDepTime\",\"CRSArrTime\",\"DepDelay\",\"Distance\" and \"TaxiOut\" columns -----------------------------------------------")
    imputer.setInputCols(numColsMean).setOutputCols(numColsMean).setStrategy("mean")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We change the value of "DepTime" and "CRSArrTime" to strings containing values such as morning, night... in order to apply one hot encoder more efficiently
    println("--------------------------------- We change the value of \"DepTime\" and \"CRSArrTime\" -----------------------------------------------")
    df = df.withColumn("DepTime", replaceTimeWithDayPart(col("DepTime")))
    df = df.withColumn("CRSArrTime", replaceTimeWithDayPart(col("CRSArrTime")))
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We check for null values in the categorical columns and swap them with "unknown"
    println("--------------------------------- We check for null values in the categorical columns and swap them with \"unknown\" -----------------------------------------------")
    val catColsDf = Array("tailNum", "Dest", "Origin", "type", "engine_type", "aircraft_type", "model", "issue_date", "manufacturer")
    for (i <- 0 until catColsDf.length) {
      val column = catColsDf(i)
      df = df.withColumn(column, replaceNullWithUnknown(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We swap columns "Year", "Month" and "DayOfMonth" with a new column "Date" so we avoid one hot encoding the "Year" column which can have only one value
    println("--------------------------------- We swap columns \"Year\", \"month\" and \"DayOfMonth\" with a new column \"Date\" so we avoid one hot encoding the \"Year\" column which can have only one value -----------------------------------------------")
    df = df.withColumn("Year", replaceYMDWithFlightDate(col("Year"), col("Month"), col("DayOfMonth")))
    df = df.withColumnRenamed("Year", "FlightDate")
    df = df.drop("Month").drop("DayOfMonth")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We create the column "PlaneAge" from the data in "Year" and "issue_date" to then remove the column "issue_date"
    println("--------------------------------- We create the column \"PlaneAge\" from the data in \"Year\" and \"issue_date\" to then remove the column \"issue_date\" -----------------------------------------------")
    df = df.withColumnRenamed("issue_date", "PlaneAge")
    df = df.withColumn("PlaneAge", year(to_date(col("FlightDate"), "M/d/y")) - year(to_date(col("PlaneAge"), "M/d/y")))
    println("--------------------------------- Done -----------------------------------------------")
    println()
    df.show()


    //df.repartition(1).write.option("header",value = true).mode(SaveMode.Overwrite).csv("src/main/resources/output")


    // We divide the variables into numerical/continuous and categorical
    val numCols = Array("FlightNum", "DepDelay", "Distance", "TaxiOut", "PlaneAge")
    val catCols = Array("FlightDateCat", "DayOfWeekCat", "DepTimeCat", "CRSArrTimeCat", "UniqueCarrierCat", "tailNumCat", "OriginCat", "DestCat", "typeCat", "manufacturerCat", "modelCat", "aircraft_typeCat", "engine_typeCat")
    val columnsToIndex = Array("FlightDate", "DayOfWeek", "DepTime", "CRSArrTime", "UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type")
    val indexedColumns = Array("FlightDateIndexed", "DayOfWeekIndexed", "DepTimeIndexed", "CRSArrTimeIndexed", "UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed")


    // Declaration of the indexer that will transform entries to integer values
    println("--------------------------------- Declaration of the indexer that will transform entries to integer values -----------------------------------------------")
    val indexer = new StringIndexer()
      .setInputCols(columnsToIndex)
      .setOutputCols(indexedColumns)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Declaration of the one hot encoder that will process the categorical variables
    println("--------------------------------- Declaration of the one hot encoder that will process the categorical variables -----------------------------------------------")
    val ohe = new OneHotEncoder()
      .setInputCols(indexedColumns)
      .setOutputCols(catCols)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    val assCols = numCols ++ catCols

    // Declaration of the assembler that will extract the features from our variables
    println("--------------------------------- Extracting features from our data -----------------------------------------------")
    val assembler = new VectorAssembler()
      .setInputCols(assCols)
      .setOutputCol("features")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Normalizing the extracted features
    println("--------------------------------- Normalizing the extracted features -----------------------------------------------")
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We use a pipeline in order to create a sequence of run stages
    println("--------------------------------- Use of a pipeline -----------------------------------------------")
    val pipeline = new Pipeline()
      .setStages(Array(indexer, ohe, assembler, normalizer))
    df = pipeline.fit(df).transform(df)
    df.printSchema()
    println("--------------------------------- Done -----------------------------------------------")
    println()


    df = df.drop(indexedColumns:_*)
    df = df.drop(columnsToIndex:_*)
    df = df.drop(catCols:_*)
    df = df.drop(Array("FlightNum", "DepDelay", "Distance", "TaxiOut", "PlaneAge", "features"):_*)
    df.show()


    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println("----------------------------------------------------------------------------- DATA MODELING ----------------------------------------------------------------------------")
    println("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    println()


    // In order to set up the models, 3-fold cross validation has been applied
    // After fitting the best model for all the algorithms, the models have been evaluated
    // with a test set for each FSS

    // Selectors have not been considered for cross validation since we want
    // to see and compare the performance for all the different FSS


    val selectorNumTopFeatures = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(3000)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selectorPercentile = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("percentile")
      .setSelectionThreshold(0.7)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selectorFalsePositiveRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("fpr")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selectorFalseDiscoveryRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("fdr")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selectorFamilywiseErrorRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("fwe")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val row = df.select("normFeatures").head
    val vector = row(0).asInstanceOf[SparseVector]
    println(s"Number of features without FSS: ${vector.size}")

    println("Performing FSS selection - numTopFeatures")
    val ntf = selectorNumTopFeatures.fit(df)
    val dfNtf = ntf.transform(df)
    println(s"Number of features after applying numTopFeatures FSS: ${ntf.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - percentile")
    val prc = selectorPercentile.fit(df)
    val dfPrc = prc.transform(df)
    println(s"Number of features after applying percentile FSS: ${prc.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - false positive rate")
    val fpr = selectorFalsePositiveRate.fit(df)
    val dfFpr = fpr.transform(df)
    println(s"Number of features after applying false positive rate FSS: ${fpr.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - false discovery rate")
    val fdr = selectorFalseDiscoveryRate.fit(df)
    val dfFdr = fdr.transform(df)
    println(s"Number of features after applying false discovery rate FSS: ${fdr.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - family-wise error rate")
    val fwe = selectorFamilywiseErrorRate.fit(df)
    val dfFwe = fwe.transform(df)
    println(s"Number of features after applying family-wise error rate FSS: ${fwe.selectedFeatures.length}")
    println("Done")


    val Array(trainingDataNtf, testDataNtf) = dfNtf.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingDataPrc, testDataPrc) = dfPrc.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingDataFpr, testDataFpr) = dfFpr.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingDataFdr, testDataFdr) = dfFdr.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingDataFwe, testDataFwe) = dfFwe.randomSplit(Array(0.7, 0.3), 10)


    println("----------------------------------------------------------------------------- LINEAR REGRESSION ----------------------------------------------------------------------------")


    // We create a linear regression learning algorithm
    val linearRegression = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("selectedFeatures")
      .setPredictionCol("predictionLR")


    // We define a grid of hyperparameter values to search over
    val lrParamGrid = new ParamGridBuilder()
      .addGrid(linearRegression.regParam, Array(0.1, 0.01, 0.001))
      .addGrid(linearRegression.elasticNetParam, Array(0.25, 0.5, 0.75))
      .addGrid(linearRegression.maxIter, Array(100, 200, 300))
      .build()


    // We create a regression evaluator for using the R Squared metric
    val lrEvaluatorR2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("r2")


    // We create a regression evaluator for using the Root Mean Squared Error metric
    val lrEvaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("rmse")


    // We define a 5-fold cross-validator
    val lrCrossValidator = new CrossValidator()
      .setEstimator(linearRegression)
      .setEvaluator(lrEvaluatorRMSE)
      .setEstimatorParamMaps(lrParamGrid)
      .setNumFolds(5)


    // We train and tune the model using k-fold cross validation
    // to after that use the best model to make predictions on the test data
    // to then evaluate the predictions using the chosen evaluation metric
    val lrModelNtf = lrCrossValidator.fit(trainingDataNtf)
    println("Model parameters - NumTopFeatures:")
    println(lrModelNtf.bestModel.extractParamMap())
    val lrPredictionsNtf = lrModelNtf.transform(testDataNtf)
    println("ArrDelay VS predictionLR - NumTopFeatures:")
    lrPredictionsNtf.select("ArrDelay", "predictionLR").show(10, false)
    println("--------------------------------- LR: Root Mean Squared Error - NumTopFeatures -----------------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsNtf))
    println("--------------------------------- LR: Coefficient of Determination (R2) - NumTopFeatures -----------------------------------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsNtf))


    val lrModelPrc = lrCrossValidator.fit(trainingDataPrc)
    println("Model parameters - Percentile:")
    println(lrModelPrc.bestModel.extractParamMap())
    val lrPredictionsPrc = lrModelPrc.transform(testDataPrc)
    println("ArrDelay VS predictionLR - Percentile:")
    lrPredictionsPrc.select("ArrDelay", "predictionLR").show(10, false)
    println("--------------------------------- LR: Root Mean Squared Error - Percentile -----------------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsPrc))
    println("--------------------------------- LR: Coefficient of Determination (R2) - Percentile -----------------------------------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsPrc))


    val lrModelFpr = lrCrossValidator.fit(trainingDataFpr)
    println("Model parameters - False Positive Rate:")
    println(lrModelFpr.bestModel.extractParamMap())
    val lrPredictionsFpr = lrModelFpr.transform(testDataFpr)
    println("ArrDelay VS predictionLR - False Positive Rate:")
    lrPredictionsFpr.select("ArrDelay", "predictionLR").show(10, false)
    println("--------------------------------- LR: Root Mean Squared Error - False Positive Rate -----------------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFpr))
    println("--------------------------------- LR: Coefficient of Determination (R2) - False Positive Rate -----------------------------------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsFpr))


    val lrModelFdr = lrCrossValidator.fit(trainingDataFdr)
    println("Model parameters - False Discovery Rate:")
    println(lrModelFdr.bestModel.extractParamMap())
    val lrPredictionsFdr = lrModelFdr.transform(testDataFdr)
    println("ArrDelay VS predictionLR - False Discovery Rate:")
    lrPredictionsFdr.select("ArrDelay", "predictionLR").show(10, false)
    println("--------------------------------- LR: Root Mean Squared Error - False Discovery Rate -----------------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFdr))
    println("--------------------------------- LR: Coefficient of Determination (R2) - False Discovery Rate -----------------------------------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsFdr))


    val lrModelFwe = lrCrossValidator.fit(trainingDataFwe)
    println("Model parameters - Family-wise Error Rate:")
    println(lrModelFwe.bestModel.extractParamMap())
    val lrPredictionsFwe = lrModelFwe.transform(testDataFwe)
    println("ArrDelay VS predictionLR - Family-wise Error Rate:")
    lrPredictionsFwe.select("ArrDelay", "predictionLR").show(10, false)
    println("--------------------------------- LR: Root Mean Squared Error - Family-wise Error Rate-----------------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFwe))
    println("--------------------------------- LR: Coefficient of Determination (R2) - Family-wise Error Rate -----------------------------------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsFwe))

/*
    println("----------------------------------------------------------------------------- DECISION TREE REGRESSOR ----------------------------------------------------------------------------")

    // We create a decision tree regressor algorithm
    val decisionTree = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("selectedFeatures")
      .setPredictionCol("predictionDTR")


    // We define a grid of hyperparameter values to search over
    val dtrParamGrid = new ParamGridBuilder()
      .addGrid(decisionTree.maxDepth, Array(3, 5, 10))
      .addGrid(decisionTree.maxBins, Array(16, 32, 64))
      .build()


    // We create a regression evaluator for using the R Squared metric
    val dtrEvaluatorR2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("r2")


    // We create a regression evaluator for using the Root Mean Squared Error metric
    val dtrEvaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("rmse")


    // We define a 5-fold cross-validator
    val dtrCrossValidator = new CrossValidator()
      .setEstimator(decisionTree)
      .setEvaluator(dtrEvaluatorRMSE)
      .setEstimatorParamMaps(dtrParamGrid)
      .setNumFolds(5)

    // We train and tune the model using k-fold cross validation
    // to after that use the best model to make predictions on the test data
    // to then evaluate the predictions using the chosen evaluation metric
    val dtrModelNtf = dtrCrossValidator.fit(trainingDataNtf)
    println("Model parameters - NumTopFeatures:")
    println(dtrModelNtf.bestModel.extractParamMap())
    val dtrPredictionsNtf = dtrModelNtf.transform(testDataNtf)
    println("ArrDelay VS predictionDTR - NumTopFeatures:")
    dtrPredictionsNtf.select("ArrDelay", "predictionDTR").show(10, false)
    println("--------------------------------- DTR: Root Mean Squared Error - NumTopFeatures -----------------------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsNtf))
    println("--------------------------------- DTR: Coefficient of Determination (R2) - NumTopFeatures -----------------------------------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsNtf))


    val dtrModelPrc = dtrCrossValidator.fit(trainingDataPrc)
    println("Model parameters - Percentile:")
    println(dtrModelPrc.bestModel.extractParamMap())
    val dtrPredictionsPrc = dtrModelPrc.transform(testDataPrc)
    println("ArrDelay VS predictionDTR - Percentile:")
    dtrPredictionsPrc.select("ArrDelay", "predictionDTR").show(10, false)
    println("--------------------------------- DTR: Root Mean Squared Error - Percentile -----------------------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsPrc))
    println("--------------------------------- DTR: Coefficient of Determination (R2) - Percentile -----------------------------------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsPrc))


    val dtrModelFpr = dtrCrossValidator.fit(trainingDataFpr)
    println("Model parameters - False Positive Rate:")
    println(dtrModelFpr.bestModel.extractParamMap())
    val dtrPredictionsFpr = dtrModelFpr.transform(testDataFpr)
    println("ArrDelay VS predictionDTR - False Positive Rate:")
    dtrPredictionsFpr.select("ArrDelay", "predictionDTR").show(10, false)
    println("--------------------------------- DTR: Root Mean Squared Error - False Positive Rate -----------------------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFpr))
    println("--------------------------------- DTR: Coefficient of Determination (R2) - False Positive Rate -----------------------------------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFpr))


    val dtrModelFdr = dtrCrossValidator.fit(trainingDataFdr)
    println("Model parameters - False Discovery Rate:")
    println(dtrModelFdr.bestModel.extractParamMap())
    val dtrPredictionsFdr = dtrModelFdr.transform(testDataFdr)
    println("ArrDelay VS predictionDTR - False Discovery Rate:")
    dtrPredictionsFdr.select("ArrDelay", "predictionDTR").show(10, false)
    println("--------------------------------- DTR: Root Mean Squared Error - False Discovery Rate -----------------------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFdr))
    println("--------------------------------- DTR: Coefficient of Determination (R2) - False Discovery Rate -----------------------------------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFdr))


    val dtrModelFwe = dtrCrossValidator.fit(trainingDataFwe)
    println("Model parameters - Family-wise Error Rate:")
    println(dtrModelFwe.bestModel.extractParamMap())
    val dtrPredictionsFwe = dtrModelFwe.transform(testDataFwe)
    println("ArrDelay VS predictionDTR - Family-wise Error Rate:")
    dtrPredictionsFwe.select("ArrDelay", "predictionDTR").show(10, false)
    println("--------------------------------- DTR: Root Mean Squared Error - Family-wise Error Rate-----------------------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFwe))
    println("--------------------------------- DTR: Coefficient of Determination (R2) - Family-wise Error Rate -----------------------------------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFwe))
*/

    //-------------------LinearRegression-----------------------------------------
    /*
    val lr = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("selectedFeatures")
      .setPredictionCol("predictionLR")
      .setMaxIter(10)

    val lr_paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.elasticNetParam, Array(1, 0.8, 0.5))
      .build()

    val lr_evaluator_rmse = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("rmse")

    val lr_evaluator_r2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("r2")

    val lr_cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(lr_evaluator_rmse)
      .setEstimatorParamMaps(lr_paramGrid)
      .setNumFolds(3)
      .setParallelism(3)

    println("-------------------------Linear Regression FPR-------------------------")
    val lr_model_fpr = lr_cv.fit(trainingData_fpr)
    println("Model parameters:")
    println(lr_model_fpr.bestModel.extractParamMap())
    val lr_predictions_fpr = lr_model_fpr.transform(testData_fpr)
    println("ArrDelay VS predictionLR:")
    lr_predictions_fpr.select("ArrDelay", "predictionLR").show(10, false)
    println(s"Root Mean Squared Error = ${lr_evaluator_rmse.evaluate(lr_predictions_fpr)}")
    println(s"R-Squared = ${lr_evaluator_r2.evaluate(lr_predictions_fpr)}")


    //-------------------DecisionTreeRegression-----------------------------------------
    val dtr = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("selectedFeatures")
      .setPredictionCol("predictionDTR")

    val dtr_evaluator_rmse = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("rmse")

    val dtr_evaluator_r2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("r2")

    val dtr_cv = new CrossValidator()
      .setEstimator(dtr)
      .setEvaluator(dtr_evaluator_rmse)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(3)
      .setParallelism(3)

    println("-------------------------Decision Tree Regression NTF-------------------------")
    val dtr_model_ntf = dtr_cv.fit(trainingDataNtf)
    println("Model parameters:")
    println(dtr_model_ntf.bestModel.extractParamMap())
    val dtr_predictions_ntf = dtr_model_ntf.transform(testData_ntf)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_ntf.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_ntf)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_ntf)}")

    println("-------------------------Decision Tree Regression PRC-------------------------")
    val dtr_model_prc = dtr_cv.fit(trainingDataPrc)
    println("Model parameters:")
    println(dtr_model_prc.bestModel.extractParamMap())
    val dtr_predictions_prc = dtr_model_prc.transform(testData_prc)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_prc.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_prc)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_prc)}")

    println("-------------------------Decision Tree Regression FPR-------------------------")
    val dtr_model_fpr = dtr_cv.fit(trainingDataFpr)
    println("Model parameters:")
    println(dtr_model_fpr.bestModel.extractParamMap())
    val dtr_predictions_fpr = dtr_model_fpr.transform(testData_fpr)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fpr.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fpr)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fpr)}")

    println("-------------------------Decision Tree Regression FDR-------------------------")
    val dtr_model_fdr = dtr_cv.fit(trainingDataFdr)
    println("Model parameters:")
    println(dtr_model_fdr.bestModel.extractParamMap())
    val dtr_predictions_fdr = dtr_model_fdr.transform(testData_fdr)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fdr.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fdr)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fdr)}")

    println("-------------------------Decision Tree Regression FWE-------------------------")
    val dtr_model_fwe = dtr_cv.fit(trainingDataFwe)
    println("Model parameters:")
    println(dtr_model_fwe.bestModel.extractParamMap())
    val dtr_predictions_fwe = dtr_model_fwe.transform(testData_fwe)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fwe.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fwe)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fwe)}")
    */


    // Summary table with RMSE and R2 measures for all the trained, validated and evaluated models
    // R2 measures the variability of the dependent variable (ArrDelay) that is explained by the predictors (must be independent variables)
    // RMSE measures the differences between predicted values by the model and the actual values.
   /* println("--------------------------------- Summary of the performance of the models -----------------------------------------------")
    val summaryDF = Seq(
      ("DECISION TREE REGRESSOR - Num Top Features Selection", dtrEvaluatorRMSE.evaluate(dtrPredictionsNtf), dtrEvaluatorR2.evaluate(dtrPredictionsNtf)),
      ("DECISION TREE REGRESSOR - Percentile Selection", dtrEvaluatorRMSE.evaluate(dtrPredictionsPrc), dtrEvaluatorR2.evaluate(dtrPredictionsPrc)),
      ("DECISION TREE REGRESSOR - False Positive Rate Selection", dtrEvaluatorRMSE.evaluate(dtrPredictionsFpr), dtrEvaluatorR2.evaluate(dtrPredictionsFpr)),
      ("DECISION TREE REGRESSOR - False Discovery Rate Selection", dtrEvaluatorRMSE.evaluate(dtrPredictionsFdr), dtrEvaluatorR2.evaluate(dtrPredictionsFdr)),
      ("DECISION TREE REGRESSOR - Family-Wise Error Rate Selection", dtrEvaluatorRMSE.evaluate(dtrPredictionsFwe), dtrEvaluatorR2.evaluate(dtrPredictionsFwe)))
      .toDF("Algorithm", "RMSE", "R2")

    summaryDF.show(false) */

//    println("--------------------------------- Separate data in train and test -----------------------------------------------")
//    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), 10)
//    println("--------------------------------- Done -----------------------------------------------")
//    println()
//
//
//    println("--------------------------------- LinearRegression Declaration -----------------------------------------------")
//    val linearRegression = new LinearRegression()
//      .setFeaturesCol("normFeatures")
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionLR")
//      .setMaxIter(10)
//
//    val linearRegression_paramGrid = new ParamGridBuilder()
//      .addGrid(linearRegression.regParam, Array(0.1, 0.01))
//      .addGrid(linearRegression.elasticNetParam, Array(1, 0.8, 0.5))
//      .build()
//
//    val linearRegression_evaluator_rmse = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionLR")
//      .setMetricName("rmse")
//
//    val linearRegression_evaluator_r2 = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionLR")
//      .setMetricName("r2")
//
//    val linearRegression_cv = new CrossValidator()
//      .setEstimator(linearRegression)
//      .setEvaluator(linearRegression_evaluator_rmse)
//      .setEstimatorParamMaps(linearRegression_paramGrid)
//      .setNumFolds(3)
//      .setParallelism(3)
//
//    println("--------------------------------- Done -----------------------------------------------")
//    println()
//
//    println("--------------------------------- LinearRegression Training -----------------------------------------------")
//    val linearRegressionModel = linearRegression_cv.fit(trainingData)
//
//    println("Model parameters:")
//    println(linearRegressionModel.bestModel.extractParamMap())
//    val linearRegression_predictions = linearRegressionModel.transform(testData)
//    println("ArrDelay VS predictionLR:")
//    linearRegression_predictions.select("ArrDelay", "predictionLR").show(10, false)
//    println(s"Root Mean Squared Error = ${linearRegression_evaluator_rmse.evaluate(linearRegression_predictions)}")
//    println(s"R-Squared = ${linearRegression_evaluator_r2.evaluate(linearRegression_predictions)}")
//
//    //-------------------DecisionTreeRegression-----------------------------------------
//    val decisionTreeRegression = new DecisionTreeRegressor()
//      .setLabelCol("ArrDelay")
//      .setFeaturesCol("normFeatures")
//      .setPredictionCol("predictionDTR")
//
//    val decisionTreeRegression_evaluator_rmse = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionDTR")
//      .setMetricName("rmse")
//
//    val decisionTreeRegression_evaluator_r2 = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionDTR")
//      .setMetricName("r2")
//
//    val decissionTreeRegression_cv = new CrossValidator()
//      .setEstimator(decisionTreeRegression)
//      .setEvaluator(decisionTreeRegression_evaluator_rmse)
//      .setEstimatorParamMaps(new ParamGridBuilder().build())
//      .setNumFolds(3)
//      .setParallelism(3)
//
//
//    println("------------------------- Decision Tree Regression -------------------------")
//    val decissionTree_model = decissionTreeRegression_cv.fit(trainingData)
//    println("Model parameters:")
//    println(decissionTree_model.bestModel.extractParamMap())
//    val decisionTreeRegression_predictions = decissionTree_model.transform(testData)
//    println("ArrDelay VS predictionDTR:")
//    decisionTreeRegression_predictions.select("ArrDelay", "predictionDTR").show(10, false)
//    println(s"Root Mean Squared Error = ${decisionTreeRegression_evaluator_rmse.evaluate(decisionTreeRegression_predictions)}")
//    println(s"R-Squared = ${decisionTreeRegression_evaluator_r2.evaluate(decisionTreeRegression_predictions)}")
//
//
//    //-------------------RandomForestRegression-----------------------------------------
//    val randomForestRegressor = new RandomForestRegressor()
//      .setLabelCol("ArrDelay")
//      .setFeaturesCol("normFeatures")
//      .setPredictionCol("predictionRFR")
//
//    val randomForestRegressor_evaluator_rmse = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionRFR")
//      .setMetricName("rmse")
//
//    val randomForestRegressor_evaluator_r2 = new RegressionEvaluator()
//      .setLabelCol("ArrDelay")
//      .setPredictionCol("predictionRFR")
//      .setMetricName("r2")
//
//    val randomForestRegressor_cv = new CrossValidator()
//      .setEstimator(randomForestRegressor)
//      .setEvaluator(randomForestRegressor_evaluator_rmse)
//      .setEstimatorParamMaps(new ParamGridBuilder().build())
//      .setNumFolds(3)
//      .setParallelism(3)
//
//    println("------------------------Random Forest Regression FPR-------------------------")
//    val randomForestRegressor_model = randomForestRegressor_cv.fit(trainingData)
//    println("Model parameters:")
//    println(randomForestRegressor_model.bestModel.extractParamMap())
//    val randomForestRegressor_predictions = randomForestRegressor_model.transform(testData)
//    println("ArrDelay VS predictionRFR:")
//    randomForestRegressor_predictions.select("normFeatures", "ArrDelay", "predictionRFR").show(10, false)
//    println(s"Root Mean Squared Error = ${randomForestRegressor_evaluator_rmse.evaluate(randomForestRegressor_predictions)}")
//    println(s"R-Squared = ${randomForestRegressor_evaluator_r2.evaluate(randomForestRegressor_predictions)}")
//
//
//    println("------------------------Summary-------------------------")
//    val summaryDF = Seq(
//      ("LINEAR REGRESSION", linearRegression_evaluator_rmse.evaluate(linearRegression_predictions), linearRegression_evaluator_r2.evaluate(linearRegression_predictions)),
//      ("DECISION TREE REGRESSION", decisionTreeRegression_evaluator_rmse.evaluate(decisionTreeRegression_predictions), decisionTreeRegression_evaluator_r2.evaluate(decisionTreeRegression_predictions)),
//      ("RANDOM FOREST REGRESSION", randomForestRegressor_evaluator_rmse.evaluate(randomForestRegressor_predictions), randomForestRegressor_evaluator_r2.evaluate(randomForestRegressor_predictions)))
//      .toDF("Algorithm", "RMSE", "R2")
//
//    summaryDF.show(false)

  }
}

/*
    // We check for missing numerical values in the each column of the dataset
    println("--------------------------------- We check for missing numerical values in the each column of the dataset -----------------------------------------------")
    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      println(column)
      df.filter(col(column).contains("NA")).show()
    }
    println("--------------------------------- Done -----------------------------------------------")

    // We check for missing numerical values in the each column of the dataset
    println("--------------------------------- We check for missing numerical values in the each column of the dataset -----------------------------------------------")
    val plane_cols_mf = Array("type", "engine_type", "aircraft_type", "model", "manufacturer")
    for (i <- 0 until plane_cols_mf.length) {
      val column = plane_cols_mf(i)
      println(column)
      df.filter(col(column).contains("NA")).show()
      df.filter(column + " is NULL OR " + column + " LIKE 'None'").show()
    }
    println("--------------------------------- Done -----------------------------------------------")

    // Categorical columns from the plane-data.csv for the "mean" imputer and the "most frequent" imputer
    println("--------------------------------- Imputing unknown value for nulls in categorical variables -----------------------------------------------")
    println("--------------------------------- Done -----------------------------------------------")

    // Change the name of the columns for the sake of the code comprehension
    for (i <- 0 until df.columns.length) {
      var colName = df.columns(i)
      var newColName = df.select(colName).first.getString(0)
      df = df.withColumnRenamed(colName, newColName)
    }

    // We delete the first row, as we do not need it anymore, given the fact that we already renamed each column
    val first_row = df.first()
    df = df.filter(row => row != first_row)
 */