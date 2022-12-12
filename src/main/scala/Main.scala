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
  private val replace_null_with_unknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow" || x == "None" || x == "" || x == " ") res = "unknown"
    else res = x
    res
  }).asNondeterministic()

  private val replace_na_with_null = udf((x: String) => {
    var res = new String
    if (x == "NA") res = null
    else res = x
    res
  }).asNondeterministic()

  private val replace_time_with_dayPart = udf((x: Integer) => {
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

  private val replace_YMD_with_FlightDate = udf((year: Integer, month: Integer, day: Integer) => {
    var res = month.toString + "/" + day.toString + "/" + year.toString
    res
  }).asNondeterministic()

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Big Data: Spark Practical Work")
      .master("local[12]")
      .config("spark.driver.memory","16G")
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


    // We read the input data
    var df = spark.read.option("header", value = "true").csv("src/main/resources/2008.csv")
    //df = df.union(spark.read.option("header", value = "true").csv("src/main/resources/2007.csv"))
    var df_plane = spark.read.option("header", value = "true").csv("src/main/resources/plane-data.csv")


    // We delete the forbidden columns
    println()
    println("--------------------------------- We delete the forbidden columns -----------------------------------------------")
    val columns_to_drop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    df = df.drop(columns_to_drop:_*)
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
    df_plane = df_plane.filter("issue_date is NOT NULL AND issue_date NOT LIKE 'None'")
    df_plane = df_plane.drop("status")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We delete the plane tailnumbers that do not have any data from plane-data dataset
    println("--------------------------------- We delete the plane tailnumbers that do not have any data from plane-data dataset -----------------------------------------------")
    df_plane = df_plane.filter("type is NOT NULL AND manufacturer is NOT NULL AND model is NOT NULL AND aircraft_type is NOT NULL AND engine_type is NOT NULL AND year is NOT NULL")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Renaming column "year" to "year_produced" in df_plane dataset and eliminating it since we are not going to use it
    println("--------------------------------- Deleting column \"year\" in df_plane dataset -----------------------------------------------")
    df_plane = df_plane.withColumnRenamed("year","year_introduced")
    df_plane = df_plane.drop("year_introduced")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Join of the two datasets
    println("--------------------------------- Joining both datasets -----------------------------------------------")
    df = df.join(df_plane, "tailNum")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We check for NA values in the each column of the dataset and set them to null for the imputers to do their work
    println("--------------------------------- Checking for NA values in the dataset to set them to null -----------------------------------------------")
    for (i <- 0 until df.columns.drop(df.columns.indexOf("ArrDelay")).length) {
      val column = df.columns(i)
      df = df.withColumn(column, replace_na_with_null(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Numerical columns for "mean" imputer and "most frequent" imputer
    val num_cols_mean = Array("DepTime","CRSArrTime","DepDelay","Distance","TaxiOut")
    val num_cols_mf = Array("FlightNum", "Year","Month","DayofMonth","DayOfWeek")


    // We cast to Integer every column in order to be able to use the imputer
    println("--------------------------------- We cast to Integer every column in order to be able to use the imputer -----------------------------------------------")
    for (i <- 0 until df.columns.length){
      val colName = df.columns(i)
      if (num_cols_mean.contains(colName) || num_cols_mf.contains(colName) || colName == "ArrDelay")
        df = df.withColumn(colName,col(colName).cast(IntegerType))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "most frequent" imputer for the "Year", "Month", "DayOfMonth" and "DayOfWeek" columns
    println("--------------------------------- We apply the \"most frequent\" imputer for the \"Year\",\"Month\",\"DayofMonth\" and \"DayOfWeek\" columns -----------------------------------------------")
    val imputer = new Imputer()
      .setInputCols(num_cols_mf)
      .setOutputCols(num_cols_mf)
      .setStrategy("mode")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "mean" imputer for the rest of the numerical columns
    println("--------------------------------- We apply the \"mean\" imputer for the \"DepTime\",\"CRSDepTime\",\"CRSArrTime\",\"DepDelay\",\"Distance\" and \"TaxiOut\" columns -----------------------------------------------")
    imputer.setInputCols(num_cols_mean).setOutputCols(num_cols_mean).setStrategy("mean")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We change the value of "DepTime" and "CRSArrTime" to strings containing values such as morning, night... in order to apply one hot encoder more efficiently
    println("--------------------------------- We change the value of \"DepTime\" and \"CRSArrTime\" -----------------------------------------------")
    df = df.withColumn("DepTime", replace_time_with_dayPart(col("DepTime")))
    df = df.withColumn("CRSArrTime", replace_time_with_dayPart(col("CRSArrTime")))
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We check for null values in the categorical columns and swap them with "unknown"
    println("--------------------------------- We check for null values in the categorical columns and swap them with \"unknown\" -----------------------------------------------")
    val cat_cols_df = Array("tailNum", "Dest", "Origin", "type", "engine_type", "aircraft_type", "model", "issue_date", "manufacturer")
    for (i <- 0 until cat_cols_df.length) {
      val column = cat_cols_df(i)
      df = df.withColumn(column, replace_null_with_unknown(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We swap columns "Year", "Month" and "DayOfMonth" with a new column "Date" so we avoid one hot encoding the "Year" column which can have only one value
    println("--------------------------------- We swap columns \"Year\", \"month\" and \"DayOfMonth\" with a new column \"Date\" so we avoid one hot encoding the \"Year\" column which can have only one value -----------------------------------------------")
    df = df.withColumn("Year", replace_YMD_with_FlightDate(col("Year"), col("Month"), col("DayOfMonth")))
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
    val num_cols = Array("FlightNum", "DepDelay", "Distance", "TaxiOut", "PlaneAge")
    val cat_cols = Array("FlightDateCat", "DayOfWeekCat", "DepTimeCat", "CRSArrTimeCat", "UniqueCarrierCat", "tailNumCat", "OriginCat", "DestCat", "typeCat", "manufacturerCat", "modelCat", "aircraft_typeCat", "engine_typeCat")
    val columns_to_index = Array("FlightDate", "DayOfWeek", "DepTime", "CRSArrTime", "UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type")
    val indexed_columns = Array("FlightDateIndexed", "DayOfWeekIndexed", "DepTimeIndexed", "CRSArrTimeIndexed", "UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed")


    // Declaration of the indexer that will transform entries to integer values
    println("--------------------------------- Declaration of the indexer that will transform entries to integer values -----------------------------------------------")
    val indexer = new StringIndexer()
      .setInputCols(columns_to_index)
      .setOutputCols(indexed_columns)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Declaration of the one hot encoder that will process the categorical variables
    println("--------------------------------- Declaration of the one hot encoder that will process the categorical variables -----------------------------------------------")
    val ohe = new OneHotEncoder()
      .setInputCols(indexed_columns)
      .setOutputCols(cat_cols)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    val ass_cols = num_cols ++ cat_cols

    // Declaration of the assembler that will extract the features from our variables
    println("--------------------------------- Extracting features from our data -----------------------------------------------")
    val assembler = new VectorAssembler()
      .setInputCols(ass_cols)
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


    df = df.drop(indexed_columns:_*)
    df = df.drop(columns_to_index:_*)
    df = df.drop(cat_cols:_*)
    df = df.drop(Array("FlightNum", "DepDelay", "Distance", "TaxiOut", "PlaneAge", "features"):_*)
    df.show()


    //-------------------FSS-----------------------------------------

    // In order to set up the models, 3-fold cross validation has been applied
    // After fitting the best model for all the algorithms, the models have been evaluated
    // with a test set for each FSS

    // Selectors have not been considered for cross validation since we want
    // to see and compare the performance for all the different FSS


    val selector_numTopFeatures = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(3000)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selector_percentile = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("percentile")
      .setSelectionThreshold(0.7)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selector_falsePositiveRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("fpr")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selector_falseDiscoveryRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous") // score function -> F-value (f_regression)
      .setSelectionMode("fdr")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val selector_familywiseErrorRate = new UnivariateFeatureSelector()
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
    val ntf = selector_numTopFeatures.fit(df)
    val df_ntf = ntf.transform(df)
    println(s"Number of features after applying numTopFeatures FSS: ${ntf.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - percentile")
    val prc = selector_percentile.fit(df)
    val df_prc = prc.transform(df)
    println(s"Number of features after applying percentile FSS: ${prc.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - false positive rate")
    val fpr = selector_falsePositiveRate.fit(df)
    val df_fpr = fpr.transform(df)
    println(s"Number of features after applying false positive rate FSS: ${fpr.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - false discovery rate")
    val fdr = selector_falseDiscoveryRate.fit(df)
    val df_fdr = fdr.transform(df)
    println(s"Number of features after applying false discovery rate FSS: ${fdr.selectedFeatures.length}")
    println("Done")

    println("Performing FSS selection - family-wise error rate")
    val fwe = selector_familywiseErrorRate.fit(df)
    val df_fwe = fwe.transform(df)
    println(s"Number of features after applying family-wise error FSS: ${fwe.selectedFeatures.length}")
    println("Done")

    val Array(trainingData_ntf, testData_ntf) = df_ntf.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingData_prc, testData_prc) = df_prc.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingData_fpr, testData_fpr) = df_fpr.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingData_fdr, testData_fdr) = df_fdr.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingData_fwe, testData_fwe) = df_fwe.randomSplit(Array(0.7, 0.3), 10)

    /*
     As it is a regression problem, that is, explaining or predicting the value of a continuous variable,
     we have tried to apply many algorithms within the family of regression algorithms
     (also trying not to extend the execution too much) and then compare the
     different results obtained. Specifically, the following algorithms have been applied: Linear regression,
     Generalized linear regression, Decision tree regression and Random forest regression.
     */

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
    val dtr_model_ntf = dtr_cv.fit(trainingData_ntf)
    println("Model parameters:")
    println(dtr_model_ntf.bestModel.extractParamMap())
    val dtr_predictions_ntf = dtr_model_ntf.transform(testData_ntf)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_ntf.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_ntf)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_ntf)}")

    println("-------------------------Decision Tree Regression PRC-------------------------")
    val dtr_model_prc = dtr_cv.fit(trainingData_prc)
    println("Model parameters:")
    println(dtr_model_prc.bestModel.extractParamMap())
    val dtr_predictions_prc = dtr_model_prc.transform(testData_prc)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_prc.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_prc)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_prc)}")

    println("-------------------------Decision Tree Regression FPR-------------------------")
    val dtr_model_fpr = dtr_cv.fit(trainingData_fpr)
    println("Model parameters:")
    println(dtr_model_fpr.bestModel.extractParamMap())
    val dtr_predictions_fpr = dtr_model_fpr.transform(testData_fpr)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fpr.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fpr)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fpr)}")

    println("-------------------------Decision Tree Regression FDR-------------------------")
    val dtr_model_fdr = dtr_cv.fit(trainingData_fdr)
    println("Model parameters:")
    println(dtr_model_fdr.bestModel.extractParamMap())
    val dtr_predictions_fdr = dtr_model_fdr.transform(testData_fdr)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fdr.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fdr)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fdr)}")

    println("-------------------------Decision Tree Regression FWE-------------------------")
    val dtr_model_fwe = dtr_cv.fit(trainingData_fwe)
    println("Model parameters:")
    println(dtr_model_fwe.bestModel.extractParamMap())
    val dtr_predictions_fwe = dtr_model_fwe.transform(testData_fwe)
    println("ArrDelay VS predictionDTR:")
    dtr_predictions_fwe.select("ArrDelay", "predictionDTR").show(10, false)
    println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fwe)}")
    println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fwe)}")


    // Summary table with RMSE and R2 measures for all the trained, validated and evaluated models
    // R2 measures the variability of the dependent variable (ArrDelay) that is explained by the predictors (must be independent variables)
    // RMSE measures the differences between predicted values by the model and the actual values.
    println("------------------------Summary-------------------------")
    val summaryDF = Seq(
      ("DECISION TREE REGRESSION - Num Top Features Selection", dtr_evaluator_rmse.evaluate(dtr_predictions_ntf), dtr_evaluator_r2.evaluate(dtr_predictions_ntf)),
      ("DECISION TREE REGRESSION - Percentile Selection", dtr_evaluator_rmse.evaluate(dtr_predictions_prc), dtr_evaluator_r2.evaluate(dtr_predictions_prc)),
      ("DECISION TREE REGRESSION - False Positive Rate Selection", dtr_evaluator_rmse.evaluate(dtr_predictions_fpr), dtr_evaluator_r2.evaluate(dtr_predictions_fpr)),
      ("DECISION TREE REGRESSION - False Discovery Rate Selection", dtr_evaluator_rmse.evaluate(dtr_predictions_fdr), dtr_evaluator_r2.evaluate(dtr_predictions_fdr)),
      ("DECISION TREE REGRESSION - Family-Wise Error Rate Selection", dtr_evaluator_rmse.evaluate(dtr_predictions_fwe), dtr_evaluator_r2.evaluate(dtr_predictions_fwe)))
      .toDF("Algorithm", "RMSE", "R2")

    summaryDF.show(false)

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