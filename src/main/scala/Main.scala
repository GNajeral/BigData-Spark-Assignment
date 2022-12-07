import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, length, udf}
import org.apache.spark.sql.types.IntegerType

import java.text.SimpleDateFormat
import java.util.{Calendar, GregorianCalendar}



object Main {
  val DateFormat = new SimpleDateFormat("MM/dd/yyyy")
  val calendar = new GregorianCalendar()

  private val replace_null_with_unknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow" || x == "" || x == " ") res = "unknown"
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
    var year = calendar.get(Calendar.YEAR)
    var res = x - year
    res
  })

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL 2008 dataset")
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    var df = spark.read.option("header",value = true).csv("src/main/resources/2008.csv")
    var df_plane = spark.read.option("header",value = true).csv("src/main/resources/plane-data.csv")


    // We delete the columns we do not need
    println("--------------------------------- We delete the columns we do not need -----------------------------------------------")
    val columns_to_drop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    df = df.drop(columns_to_drop:_*)
    println("--------------------------------- Done -----------------------------------------------")


    // We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal
    println("--------------------------------- We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal -----------------------------------------------")
    df = df.filter("Cancelled == 0")
    println("--------------------------------- Done -----------------------------------------------")

    df_plane = df_plane.filter("issue_date is NOT NULL OR issue_date NOT LIKE 'None'")
    df_plane = df_plane.drop("status")


    // We delete the "CRSDepTime" column given that we already have enough information with the "DepTime" and "DepDelay" columns
    println("--------------------------------- We delete \"CRSDepTime\" -----------------------------------------------")
    df = df.drop("CRSDepTime")
    println("--------------------------------- Done -----------------------------------------------")


    // We delete the plane tailnumbers that dont have any data from plane-data dataset
    println("--------------------------------- We delete the plane tailnumbers that dont have any data from plane-data dataset -----------------------------------------------")
    df_plane = df_plane.filter("type is NOT NULL AND manufacturer is NOT NULL AND issue_date is NOT NULL AND model is NOT NULL AND aircraft_type is NOT NULL AND engine_type is NOT NULL AND year is NOT NULL")
    println("--------------------------------- Done -----------------------------------------------")


    // Renaming column "year" to "year_produced" in df_plane dataset and elimnate it
    println("--------------------------------- Renaming column \"year\" to \"year_introduced\" in df_plane dataset -----------------------------------------------")
    df_plane = df_plane.withColumnRenamed("year","year_introduced")
    df_plane = df_plane.drop("year_introduced")
    println("--------------------------------- Done -----------------------------------------------")


    // Therefore we eliminate the "CancellationCode" and "Cancelled" columns
    println("--------------------------------- Therefore we eliminate the \"CancellationCode\" and \"Cancelled\" columns -----------------------------------------------")
    df = df.drop("Cancelled","CancellationCode")
    println("--------------------------------- Done -----------------------------------------------")


    // We separate our target_variable from the rest of the dataset, saving it in a different one
    var t_col = df.select("ArrDelay")
    df = df.drop("ArrDelay")
    println("--------------------------------- Target Variable -----------------------------------------------")
    t_col.show


    // We check for null values in the "TailNum" column and swap them with "unknown"
    println("--------------------------------- We check for null values in the \"TailNum\" column and swap them with \"unknown\" -----------------------------------------------")
    spark.udf.register("replace_null_with_unknown", replace_null_with_unknown)
    spark.udf.register("replace_na_with_null", replace_na_with_null)
    spark.udf.register("replace_issueDate_with_planeAge", replace_issueDate_with_planeAge)
    println("--------------------------------- Done -----------------------------------------------")

    // Join of the two datasets
    println("--------------------------------- Joining both datasets -----------------------------------------------")
    df = df.join(df_plane, "tailNum")
    println("--------------------------------- Done -----------------------------------------------")


    // We check for NA values in the each column of the dataset and set them to null for the imputers to do their work
    println("--------------------------------- Checking for NA values in the dataset to set them to null for the imputers to do their work -----------------------------------------------")
    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      df = df.withColumn(column, replace_na_with_null(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")


    // Numerical columns for mean imputer and most frequent imputer
    println("--------------------------------- Numerical columns to be transformed: -----------------------------------------------")
    val num_cols_mean = Array("DepTime","CRSArrTime","CRSElapsedTime","DepDelay","Distance","TaxiOut")
    val num_cols_mf = Array("Year","Month","DayofMonth","DayOfWeek")


    // We cast to Integer every column in order to be able to use the imputer
    println("--------------------------------- We cast to Integer every column in order to be able to use the imputer -----------------------------------------------")
    for (i <- 0 until  df.columns.length){
      val colName = df.columns(i)
      if (num_cols_mean.contains(colName) || num_cols_mf.contains(colName))
        df = df.withColumn(colName,col(colName).cast(IntegerType))
    }
    println("--------------------------------- Done -----------------------------------------------")


    // We apply the most frequent imputer for the columns Year Month DayOfMonth and DayOfWeek
    println("--------------------------------- We apply the most frequent imputer for the columns \"Year\",\"Month\",\"DayofMonth\",\"DayOfWeek\" -----------------------------------------------")
    val imputer = new Imputer()
      .setInputCols(num_cols_mf)
      .setOutputCols(num_cols_mf)
      .setStrategy("mode")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")


    // We apply the mean imputer for the rest of the numerical columns
    println("--------------------------------- We apply the mean imputer for the columns \"DepTime\",\"CRSDepTime\",\"CRSArrTime\",\"CRSElapsedTime\",\"DepDelay\",\"Distance\",\"TaxiOut\" -----------------------------------------------")
    imputer.setInputCols(num_cols_mean).setOutputCols(num_cols_mean).setStrategy("mean")
    df = imputer.fit(df).transform(df)
    println("--------------------------------- Done -----------------------------------------------")


    // Categorical columns from the plane-data.csv for mean imputer and most frequent imputer
    println("--------------------------------- Imputing unknown value for nulls in categorical variables -----------------------------------------------")
    val cat_cols_df = Array("tailNum", "Dest", "Origin", "type", "engine_type", "aircraft_type", "model", "issue_date", "manufacturer")
    for (i <- 0 until cat_cols_df.length) {
      val column = cat_cols_df(i)
      df = df.withColumn(column, replace_null_with_unknown(col(column)))
    }
    println("--------------------------------- Done -----------------------------------------------")


    // We create the column "PlaneAge" from the data in "Year" and "issue_date", to then remove the column "issue_date"
    println("--------------------------------- We create the column \"PlaneAge\" from the data in \"Year\" and \"issue_date\", to then remove the column \"issue_date\" -----------------------------------------------")
    df = df.withColumn("issue_date", replace_issueDate_with_planeAge(col("Year"), col("issue_date")))
    df = df.withColumnRenamed("issue_date", "PlaneAge")
    println("--------------------------------- Done -----------------------------------------------")

    /*
    // We check for missing numerical values in the each column of the dataset
    println("--------------------------------- We check for missing numerical values in the each column of the dataset -----------------------------------------------")
    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      println(column)
      df.filter(col(column).contains("NA")).show()
    }
    println("--------------------------------- Done -----------------------------------------------")
*/

    println("--------------------------------- Processed Dataset -----------------------------------------------")
    df.show()

    // We check for missing numerical values in the each column of the dataset
    println("--------------------------------- We check for missing numerical values in the each column of the dataset -----------------------------------------------")
    val plane_cols_mf = Array("type", "engine_type", "aircraft_type", "model", "manufacturer")
    for (i <- 0 until plane_cols_mf.length) {
      val column = plane_cols_mf(i)
      println(column)
      //df.filter(col(column).contains("NA")).show()
      df.filter(column + " is NULL OR " + column + " LIKE 'None'").show()
    }
    println("--------------------------------- Done -----------------------------------------------")



    // Declaration of the indexer that will transform entries to integer values
    val indexer = new StringIndexer().setInputCols(Array("UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type")).setOutputCols(Array("UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed"))


    // Declaration of the one hot encoder that will process the categorical variables
    val ohe = new OneHotEncoder().setInputCols(Array("UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed")).setOutputCols(Array("UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type"))

    println("--------------------------------- Indexing Dataset -----------------------------------------------")
    df = indexer.fit(df).transform(df)
    df = df.drop("UniqueCarrier", "tailNum", "Origin", "Dest", "type", "manufacturer", "model", "aircraft_type", "engine_type")
    println("--------------------------------- Done -----------------------------------------------")


    println("--------------------------------- Applying One Hot Encoder -----------------------------------------------")
    df = ohe.fit(df).transform(df)
    df = df.drop("UniqueCarrierIndexed", "tailNumIndexed", "OriginIndexed", "DestIndexed", "typeIndexed", "manufacturerIndexed", "modelIndexed", "aircraft_typeIndexed", "engine_typeIndexed")
    println("--------------------------------- Done -----------------------------------------------")


    println("--------------------------------- Preprocessed Dataset -----------------------------------------------")
    df.show()



  }

}

//    // Change the name of the columns for the sake of the code comprehension
//    for (i <- 0 until df.columns.length) {
//      var colName = df.columns(i)
//      var newColName = df.select(colName).first.getString(0)
//      df = df.withColumnRenamed(colName, newColName)
//    }

//    // We delete the first row, as we do not need it anymore, given the fact that we already renamed each column
//    val first_row = df.first()
//    df = df.filter(row => row != first_row)
