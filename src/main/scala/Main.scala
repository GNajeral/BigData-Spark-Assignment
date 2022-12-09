import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.IntegerType

import java.text.SimpleDateFormat
import java.util.{Calendar, GregorianCalendar}



object Main {
  private val DateFormat = new SimpleDateFormat("MM/dd/yyyy")
  private val calendar = new GregorianCalendar()

  private val replace_null_with_unknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow" || x == "None" || x == "" || x == " ") res = "unknown"
    else res = x
    res
  })

  private val replace_na_with_null = udf((x: String) => {
    var res = new String
    if (x == "NA") res = null
    else res = x
    res
  })

  private val replace_issueDate_with_planeAge = udf((x: Integer, y: String) => {
    calendar.setTime(DateFormat.parse(y))
    val year = calendar.get(Calendar.YEAR)
    val res = x - year
    res
  })

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
  })

  private val replace_YMD_with_FlightDate = udf((year: Integer, month: Integer, day: Integer) => {
    var res = month.toString + "/" + day.toString + "/" + year.toString
    res
  })

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL 2008 dataset")
      .master("local[12]")
      .config("spark.driver.memory","16G")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // We read the input data
    var df = spark.read.option("header",value = true).csv("src/main/resources/2008.csv")
    var df_plane = spark.read.option("header",value = true).csv("src/main/resources/plane-data.csv")

    // We register several UDFs that are going to be used
    spark.udf.register("replace_null_with_unknown", replace_null_with_unknown)
    spark.udf.register("replace_na_with_null", replace_na_with_null)
    spark.udf.register("replace_issueDate_with_planeAge", replace_issueDate_with_planeAge)
    spark.udf.register("replace_time_with_dayPart", replace_time_with_dayPart)


    // We delete the forbidden columns
    println()
    println("--------------------------------- We delete the forbidden columns -----------------------------------------------")
    val columns_to_drop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    df = df.drop(columns_to_drop:_*)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We separate our target variable from the rest of the dataset, saving it in a different one
    println("--------------------------------- We separate the target variable -----------------------------------------------")
    val t_col = df.select("ArrDelay")
    df = df.drop("ArrDelay")
    t_col.show()
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
    println("--------------------------------- We delete \"CRSDepTime\" -----------------------------------------------")
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
    df_plane = df_plane.filter("type is NOT NULL AND manufacturer is NOT NULL AND issue_date is NOT NULL AND model is NOT NULL AND aircraft_type is NOT NULL AND engine_type is NOT NULL AND year is NOT NULL")
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
    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      df = df.withColumn(column, replace_na_with_null(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // Numerical columns for "mean" imputer and "most frequent" imputer
    val num_cols_mean = Array("DepTime","CRSArrTime","DepDelay","Distance","TaxiOut")
    val num_cols_mf = Array("Year","Month","DayofMonth","DayOfWeek")


    // We cast to Integer every column in order to be able to use the imputer
    println("--------------------------------- We cast to Integer every column in order to be able to use the imputer -----------------------------------------------")
    for (i <- 0 until df.columns.length){
      val colName = df.columns(i)
      if (num_cols_mean.contains(colName) || num_cols_mf.contains(colName))
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

    df.filter("issue_date is NULL OR issue_date LIKE 'unknown'").show()

    // We create the column "PlaneAge" from the data in "Year" and "issue_date" to then remove the column "issue_date"
    println("--------------------------------- We create the column \"PlaneAge\" from the data in \"Year\" and \"issue_date\" to then remove the column \"issue_date\" -----------------------------------------------")
    df = df.withColumn("issue_date", replace_issueDate_with_planeAge(col("Year"), col("issue_date")))
    df = df.withColumnRenamed("issue_date", "PlaneAge")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We swap columns "Year", "Month" and "DayOfMonth" with a new column "Date" so we avoid one hot encoding the "Year" column which can have only one value
    println("--------------------------------- We swap columns \"Year\", \"month\" and \"DayOfMonth\" with a new column \"Date\" so we avoid one hot encoding the \"Year\" column which can have only one value -----------------------------------------------")
    df = df.withColumn("Year", replace_YMD_with_FlightDate(col("Year"), col("Month"), col("DayOfMonth")))
    df = df.withColumnRenamed("Year", "FlightDate")
    df = df.drop("Month").drop("DayOfMonth")
    println("--------------------------------- Done -----------------------------------------------")
    println()


    println("--------------------------------- 1st Preprocessed Dataset -----------------------------------------------")
    df.show()


    //df.repartition(1).write.option("header",value = true).mode(SaveMode.Overwrite).csv("src/main/resources/output")


    val columns_to_index = Array("FlightDate", "DayOfWeek", "DepTime", "CRSArrTime", "UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type")
    val indexed_columns = Array("FlightDateIndexed", "DayOfWeekIndexed", "DepTimeIndexed", "CRSArrTimeIndexed", "UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed")
    val cat_cols = Array("FlightDateCat", "DayOfWeekCat", "DepTimeCat", "CRSArrTimeCat", "UniqueCarrierCat", "tailNumCat", "OriginCat", "DestCat", "typeCat", "manufacturerCat", "modelCat", "aircraft_typeCat", "engine_typeCat")

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


    // Indexing Dataset
    println("--------------------------------- Indexing Dataset -----------------------------------------------")
    df = indexer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()


    // We apply the One Hot Encoder to the dataset
    println("--------------------------------- Applying One Hot Encoder -----------------------------------------------")
    df = ohe.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")
    println()

    //df = df.drop(indexed_columns:_*)
    //df = df.drop(columns_to_index:_*)

    println("--------------------------------- Final Preprocessed Dataset -----------------------------------------------")
    df.show()





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